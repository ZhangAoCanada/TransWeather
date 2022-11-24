from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class SECONDFPNConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
      self.bn = nn.BatchNorm2d(out_channels)
      self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class UpsampleConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super(UpsampleConv2d, self).__init__()
      self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding)
      self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
      self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.upsample(out)
        out = self.conv2(out)
        return out



class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
    
    
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) * 0.1
        # x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU()
        self.conv = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        input = x
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv(x)

        x = input + x * 0.1
        return x


class MBConvBlockProcess(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU()
        self.conv = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        input = x
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv(x)

        # x = input + x * 0.1
        x = input + x
        return x



class MBConvBlockStage(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU()
        self.conv = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        input = x
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv(x)

        x = input + x
        return x

    
def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module
