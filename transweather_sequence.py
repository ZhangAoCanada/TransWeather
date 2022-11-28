import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import xavier_init, constant_init
from base_networks import *
from multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp16, MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from transweather_model import Transweather
from transweather_model_teacher import TransweatherTeacher



class LSTMCustom(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(LSTMCustom, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.ffn = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        assert query.shape == x.shape
        b, c, w, h = x.shape
        query = query.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)

        h_0 = Variable(torch.zeros(1, x.size(1), 256).cuda())
        c_0 = Variable(torch.zeros(1, x.size(1), 256).cuda())
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.ffn(out)

        out = out.view(b, w, h, c).permute(0, 3, 1, 2)
        return out



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



class CrossDeformableAttention(nn.Module):
    def __init__(self, dim, h, w, num_heads, num_levels, num_points):
        super(CrossDeformableAttention, self).__init__()
        # self.multiscale_deformable_attn_fn = MultiScaleDeformableAttnFunction_fp32
        self.multiscale_deformable_attn_fn = multi_scale_deformable_attn_pytorch
        self.dim = dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = self.num_points
        self.shape_tensor = torch.tensor([h, w], dtype=torch.long).repeat(num_levels, 1).cuda()
        self.sampling_locations = nn.Linear(dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self, query, value):
        assert query.shape == x.shape
        b, c, w, h = x.shape
        num_query = num_value = w * h
        query = query.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)

        # b, num_query, _ = query.shape
        # b, num_value, _ = value.shape
        assert (self.shape_tensor[:, 0] * self.shape_tensor[:, 1]).sum() == num_value
        sampling_locations = self.sampling_locations(query).view(b, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(b, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = attention_weights.view(b, num_query, self.num_heads, self.num_levels, self.num_points)
        value = self.value_proj(value)
        output = self.multiscale_deformable_attn_fn.apply(value, self.shape_tensor, sampling_locations, attention_weights)
        output = self.output_proj(output)

        out = out.view(b, w, h, c).permute(0, 3, 1, 2)
        return output



# class TransweatherSeq(Transweather):
class TransweatherSeq(TransweatherTeacher):
    def __init__(self, ckpt_path=None):
        super(TransweatherSeq, self).__init__(ckpt_path)
        self.previous_feature = None
        # self.cross_attn = CrossAttention(dim=512, num_heads=8, dropout=0.1)
        self.cross_attn = CrossAttentionFast(dim=512, num_heads=8, attn_drop=0., sr_ratio=2)
   
    def updatePreviousFeature(self, x):
        self.eval()
        with torch.no_grad():
            ### TODO: aggregate previous feature and current feature to 
            ###      get a new feature
            # self.previous_feature = concat(self.previous_feature, x[0])
            # self.previous_feature = residualbasic(self.previous_feature)
            output = x.clone()
            self.train()
            return output
    
    def sequenceAttention(self, x, reset=False):
        x_in = x[0]
        if self.previous_feature is None:
            self.previous_feature = self.updatePreviousFeature(x_in)
        output = self.cross_attn(self.previous_feature, x_in)
        self.previous_feature = self.updatePreviousFeature(output)
        return [output]

    def forward(self, x, reset=False):
        if reset:
            self.previous_feature = None

        x1 = self.Tenc(x)

        x2 = self.Tdec(x1)

        x2 = self.sequenceAttention(x2, reset=reset)

        x = self.convtail(x1, x2)

        clean = self.active(self.clean(x))

        return clean