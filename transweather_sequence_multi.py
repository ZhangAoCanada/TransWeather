from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import xavier_init, constant_init
from base_networks import *
from multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp16, MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch, MultiScaleDeformableAttention
from transweather_model import Transweather
from transweather_model_teacher import TransweatherTeacher

from typing import Sequence
from functools import partial, reduce
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_


class LSTMSeq(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(LSTMSeq, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, dim)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.h = None
        self.c = None
    
    def reset_hidden(self):
        self.h = None
        self.c = None
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)
        if self.h is None:
            self.h = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
            self.c = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
        out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        self.h.detach_()
        self.c.detach_()
        output = self.fc1_act(self.fc1(out))
        output = self.fc2(output)

        output = output.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return output
    


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, query, x):
        assert query.shape == x.shape
        b, c, w, h = x.shape
        query = query.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)

        out, _ = self.multihead_attn(query, x, x)
        out = self.ffn(out)

        out = out.view(b, w, h, c).permute(0, 3, 1, 2)
        return out



class CrossAttentionFast(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query, x):
        # assert query.shape == x.shape
        if not query.shape == x.shape:
            raise ValueError(f"query shape {query.shape} should be equal to x shape {x.shape}.")
        B, C, W, H = x.shape
        N = W * H
        query = query.permute(0, 2, 3, 1).contiguous().view(B, W*H, C)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, W*H, C)
        
        ### NOTE: processing query
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, W, H, C).permute(0, 3, 1, 2)
        return x

class DownSampleResidualBlock(nn.Module):
    def __init__(self, dim, dim_out=None, stride=2, downsample=None):
        super(DownSampleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim_out or dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim_out or dim)

        self.downsample = nn.Conv2d(dim, dim_out or dim, kernel_size=1, stride=stride, bias=False) if downsample is None else downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CrossDeformableAttention(nn.Module):
    def __init__(self, dim, num_heads, num_levels, num_points, attn_drop=0.):
        super(CrossDeformableAttention, self).__init__()
        self.multiscale_deformable_attn_fn = MultiScaleDeformableAttention(dim, num_heads, num_levels, num_points, dropout=attn_drop)
        self.dim = dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.output_proj = nn.Linear(dim, dim)
        
    def init_weights(self):
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self, query, value):
        assert query.shape == value.shape
        x_in = value
        b, c, w, h = value.shape
        num_query = num_value = w * h
        query = query.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)
        value = value.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)

        spatial_shapes = torch.tensor([h, w], dtype=torch.long).repeat(self.num_levels, 1).cuda()

        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # create meshgrid with shape (H, W, 2)
        meshgrid = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w)]), dim=-1).float().cuda()
        meshgrid = meshgrid / torch.tensor([h - 1, w - 1], dtype=torch.float).cuda()
        meshgrid = meshgrid.view(1, h, w, 1, 2).repeat(b, 1, 1, self.num_levels, 1).view(b, num_query, self.num_levels, 2)

        spatial_shapes_for_start = [(h, w)]
        spatial_shapes_for_start = torch.as_tensor(spatial_shapes_for_start, dtype=torch.long).cuda()
        level_start_index = torch.cat((spatial_shapes_for_start.new_zeros((1,)), spatial_shapes_for_start.prod(1).cumsum(0)[:-1]))

        output = self.multiscale_deformable_attn_fn(
                        query=query, 
                        value=value, 
                        reference_points=meshgrid, 
                        spatial_shapes=spatial_shapes, 
                        level_start_index=level_start_index
                        )
        output = output.permute(1, 0, 2)
        output = self.output_proj(output)

        output = output.view(b, w, h, c).permute(0, 3, 1, 2)

        ### TODO: residual connection ###
        output = output + x_in

        return output


############################################################################
############################################################################
######################### NOTE: from MetaFormer ############################
############################################################################
############################################################################
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class SpatialFc(nn.Module):
    """SpatialFc module that take features with shape of (B,C,*) as input.
    """
    def __init__(
        self, spatial_shape=[14, 14], **kwargs, 
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        N = reduce(lambda x, y: x * y, spatial_shape)
        self.fc = nn.Linear(N, N, bias=False)

    def forward(self, x):
        # input shape like [B, C, H, W]
        shape = x.shape
        x = torch.flatten(x, start_dim=2) # [B, C, H*W]
        x = self.fc(x) # [B, C, H*W]
        x = x.reshape(*shape) # [B, C, H, W]
        return x

class AddPositionEmb(nn.Module):
    """Module to add position embedding to input features
    """
    def __init__(
        self, dim=384, spatial_shape=[14, 14],
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        if len(spatial_shape) == 1:
            embed_shape = list(spatial_shape) + [dim]
        else:
            embed_shape = [dim] + list(spatial_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))
    def forward(self, x):
        return x+self.pos_embed
############################################################################
############################################################################
######################### NOTE: from MetaFormer ############################
############################################################################
############################################################################



class DeepSequence(nn.Module):
    def __init__(self, dim, num_depth, num_prev):
        super(DeepSequence, self).__init__()
        self.dim = dim
        self.num_depth = num_depth
        self.num_prev = num_prev
        self.prev_queue = []
        self.layers = nn.ModuleList(
                            [CrossDeformableAttention(
                                        dim=512, 
                                        num_heads=8, 
                                        num_levels=1, 
                                        num_points=4, 
                                        attn_drop=0.
                                        ) for _ in range(self.num_depth)]
                            )
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def checkQueue(self):
        while len(self.prev_queue) > self.num_prev:
            self.prev_queue.pop(0)
    
    def getPrev(self, x):
        if len(self.prev_queue) == 0:
            return x

        prev = self.prev_queue[0]
        for i in range(len(self.prev_queue) - 1):
            prev_output = self.forwardProcess(self.prev_queue[i+1], prev)
            prev += prev_output

        # prev = torch.zeros_like(x)
        # for i in range(len(self.prev_queue)):
        #     prev += self.forwardProcess(self.prev_queue[i], x)

        return prev
    
    def forwardProcess(self, prev, x):
        self.eval()
        with torch.no_grad():
            # prev = prev + self.pos_embed
            x = x + self.pos_embed
            for layer in self.layers:
                # x = layer(prev, x)
                prev = layer(prev, x)
            x = prev
        self.train()
        self.prev_queue.append(x)
        self.checkQueue()
        return x
    
    def forward(self, x, reset):
        x_in = x
        if reset:
            self.prev_queue = []
        prev = self.getPrev(x)
        # prev = prev + self.pos_embed
        x = x + self.pos_embed
        for layer in self.layers:
            # x = layer(prev, x)
            prev = layer(prev, x)
        x = prev
        x = x + x_in
        return x


# class TransweatherSeq(Transweather):
class TransweatherSeq(TransweatherTeacher):
    def __init__(self, ckpt_path=None, num_repeat=1):
        super(TransweatherSeq, self).__init__(ckpt_path)
        self.num_repeat = 1
        self.deep_seq_list = nn.ModuleList(
                                [DeepSequence(
                                            dim=512, 
                                            num_depth=6, 
                                            num_prev=10
                                            ) for _ in range(self.num_repeat)]
                                )

    def forward(self, x, reset=False):
        x1 = self.Tenc(x)
        x2 = self.Tdec(x1)

        x_in = x2[0]
        for deep_seq in self.deep_seq_list:
            x_in = deep_seq(x_in, reset)

        x2 = [x_in]
        x = self.convtail(x1, x2)
        clean = self.active(self.clean(x))

        return clean
    
    # only train self.cross_attn
    def train(self, mode=True):
        super(TransweatherSeq, self).train(False)
        self.Tenc.train(False)
        self.Tdec.train(False)
        self.convtail.train(False)
        self.clean.train(False)
        for layer in self.deep_seq_list:
            layer.train(mode)


