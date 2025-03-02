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


























# class CrossDeformableAttention(nn.Module):
#     def __init__(self, dim, num_heads, num_levels, num_points, attn_drop=0.):
#         super(CrossDeformableAttention, self).__init__()
#         self.multiscale_deformable_attn_fn = MultiScaleDeformableAttention(dim, num_heads, num_levels, num_points, dropout=attn_drop)
#         self.dim = dim
#         self.num_heads = num_heads
#         self.num_levels = num_levels
#         self.num_points = num_points
#         self.output_proj = nn.Linear(dim, dim)
        
#         ### NOTE: extra ###
#         self.previous_features = None

#         ### NOTE: for concat and process ###
#         self.middle_forward = DownSampleResidualBlock(dim=512 * 2, dim_out=512, stride=1)
    
#     def init_weights(self):
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
#     ### NOTE: extra ###
#     def reset_previous_features(self):
#         self.previous_features = None
    
#     ### NOTE: extra ###
#     def update_previous_features(self, x):
#         self.eval()
#         with torch.no_grad():
#             self.previous_features = x.clone()
#         self.train()

#     ### NOTE: for concat and process ###
#     def process_previous_feature(self, x):
#         output = torch.concat([self.previous_features, x], dim=1)
#         output = self.middle_forward(output)
#         return output
    
#     def forward(self, value):
#         ### NOTE: extra ###
#         if self.previous_features is None:
#             self.update_previous_features(value)
#         ### NOTE: for concat and process ###
#         self.previous_features = self.process_previous_feature(value) 
#         query = self.previous_features

#         assert query.shape == value.shape
#         x_in = value
#         b, c, w, h = value.shape
#         num_query = num_value = w * h
#         query = query.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)
#         value = value.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)

#         spatial_shapes = torch.tensor([h, w], dtype=torch.long).repeat(self.num_levels, 1).cuda()

#         query = query.permute(1, 0, 2)
#         value = value.permute(1, 0, 2)

#         # create meshgrid with shape (H, W, 2)
#         meshgrid = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w)]), dim=-1).float().cuda()
#         meshgrid = meshgrid / torch.tensor([h - 1, w - 1], dtype=torch.float).cuda()
#         meshgrid = meshgrid.view(1, h, w, 1, 2).repeat(b, 1, 1, self.num_levels, 1).view(b, num_query, self.num_levels, 2)

#         spatial_shapes_for_start = [(h, w)]
#         spatial_shapes_for_start = torch.as_tensor(spatial_shapes_for_start, dtype=torch.long).cuda()
#         level_start_index = torch.cat((spatial_shapes_for_start.new_zeros((1,)), spatial_shapes_for_start.prod(1).cumsum(0)[:-1]))

#         output = self.multiscale_deformable_attn_fn(
#                         query=query, 
#                         value=value, 
#                         reference_points=meshgrid, 
#                         spatial_shapes=spatial_shapes, 
#                         level_start_index=level_start_index
#                         )
#         output = output.permute(1, 0, 2)
#         output = self.output_proj(output)

#         output = output.view(b, w, h, c).permute(0, 3, 1, 2)

#         ### NOTE: extra ###
#         self.update_previous_features(output)

#         ### TODO: residual connection ###
#         output = output + x_in

#         return output


# # class TransweatherSeq(Transweather):
# class TransweatherSeq(TransweatherTeacher):
#     def __init__(self, seq_depth=1, ckpt_path=None):
#         super(TransweatherSeq, self).__init__(ckpt_path)
#         self.seq_depth = 6
#         # self.layers = nn.ModuleList([CrossAttention(dim=512, num_heads=8, dropout=0.1) for _ in range(self.seq_depth)])
#         # self.layers = nn.ModuleList([CrossAttentionFast(dim=512, num_heads=8, attn_drop=0., sr_ratio=2) for _ in range(self.seq_depth)])

#         # self.layers = nn.ModuleList([CrossDeformableAttention(dim=512, num_heads=8, num_levels=1, num_points=4, attn_drop=0.) for _ in range(self.seq_depth)])

#         self.lstm = LSTMSeq(dim=512, hidden_dim=512, num_layers=self.seq_depth)
   
#     def updatePreviousFeature(self, x):
#         self.eval()
#         with torch.no_grad():
#             output = x.clone()
#             self.train()
#             return output

#     # def crossAttnSelf(self, x, reset=False):
#     #     x = x[0]
#     #     for layer in self.layers:
#     #         if reset:
#     #             layer.reset_previous_features()
#     #         x = layer(x)
#     #     return [x]
    
#     def lstmAttn(self, x, reset=False):
#         x = x[0]
#         if reset:
#             self.lstm.reset_hidden()
#         x = self.lstm(x)
#         return [x]

#     def forward(self, x, reset=False):

#         x1 = self.Tenc(x)
#         x2 = self.Tdec(x1)

#         x2 = self.lstmAttn(x2, reset=reset)
#         # x2 = self.crossAttnSelf(x2, reset=reset)

#         x = self.convtail(x1, x2)
#         clean = self.active(self.clean(x))

#         return clean
    
#     ### NOTE: only train self.cross_attn ###
#     def train(self, mode=True):
#         # super(TransweatherSeq, self).train(False)
#         self.Tenc.train(False)
#         self.Tdec.train(False)
#         self.convtail.train(False)
#         self.clean.train(False)
#         # for layer in self.layers:
#         #     layer.train(mode)
#         self.lstm.train(mode)



































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


# class TransweatherSeq(Transweather):
class TransweatherSeq(TransweatherTeacher):
    def __init__(self, ckpt_path=None):
        super(TransweatherSeq, self).__init__(ckpt_path)
        self.seq_depth = 6
        self.previous_feature = None
        # self.cross_attn = CrossAttention(dim=512, num_heads=8, dropout=0.1)
        # self.cross_attn = CrossAttentionFast(dim=512, num_heads=8, attn_drop=0., sr_ratio=2)

        # self.cross_attn = CrossDeformableAttention(dim=512, num_heads=8, num_levels=1, num_points=4, attn_drop=0.)
        self.layers = nn.ModuleList([CrossDeformableAttention(dim=512, num_heads=8, num_levels=1, num_points=4, attn_drop=0.) for _ in range(self.seq_depth)])

        ### NOTE: for concat and process ###
        self.middle_forward = DownSampleResidualBlock(dim=512 * 2, dim_out=512, stride=1)

    def updatePreviousFeature(self, x):
        self.eval()
        with torch.no_grad():
            output = x.clone()
            self.train()
        return output
    
    ### NOTE: for concat and process ###
    def processPreviousFeature(self, x):
        output = torch.concat([self.previous_feature, x], dim=1)
        output = self.middle_forward(output)
        return output

    def sequenceAttention(self, x, reset=False):
        x = x[0]
        if self.previous_feature is None:
            self.previous_feature = self.updatePreviousFeature(x)

        ### NOTE: for concat and process ###
        self.previous_feature = self.processPreviousFeature(x)

        # output = self.cross_attn(self.previous_feature, x_in)
        for layer in self.layers:
            x = layer(self.previous_feature, x)

        self.previous_feature = self.updatePreviousFeature(x)
        return [x]

    def forward(self, x, reset=False):
        if reset:
            self.previous_feature = None

        x1 = self.Tenc(x)
        x2 = self.Tdec(x1)

        x2 = self.sequenceAttention(x2, reset=reset)

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
        # self.cross_attn.train(mode)
        for layer in self.layers:
            layer.train(mode)


