import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import os
from utils import *
__all__ = ['MixUNet']

import timm
from timm.models.layers import DropPath, trunc_normal_
import types
import math

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.drop = nn.Dropout(drop)

        self.dwconv_ = DWConv(2 * out_features, out_features)

        self.shift_size = shift_size
        self.pad = shift_size // 2
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
    def shift(x, dim):
        x = F.pad(x, "constant", 0)
        x = torch.chunk(x, shift_size, 1)
        x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
        x = torch.cat(x, 1)
        return x[:, :, pad:-pad, pad:-pad]
    def forward(self, x, H, W):
        y = x
        B, N, C = x.shape

        # First branch shift W
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_w = x_s.transpose(1, 2)

        x = self.fc1(x_shift_w)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        # First branch shift H
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_h = x_s.transpose(1, 2)

        x = self.fc2(x_shift_h)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        ###############################################
        # Second branch shift H
        yn = y.transpose(1, 2).view(B, C, H, W).contiguous()
        yn = F.pad(yn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        ys = torch.chunk(yn, self.shift_size, 1)
        y_shift = [torch.roll(y_c, shift, 3) for y_c, shift in zip(ys, range(-self.pad, self.pad + 1))]
        y_cat = torch.cat(y_shift, 1)
        y_cat = torch.narrow(y_cat, 2, self.pad, H)
        y_s = torch.narrow(y_cat, 3, self.pad, W)
        y_s = y_s.reshape(B, C, H * W).contiguous()
        y_shift_c = y_s.transpose(1, 2)

        y = self.fc1(y_shift_c)

        y = self.dwconv(y, H, W)
        y = self.act(y)
        y = self.drop(y)

        # Second branch shift W
        yn = y.transpose(1, 2).view(B, C, H, W).contiguous()
        yn = F.pad(yn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        ys = torch.chunk(yn, self.shift_size, 1)
        y_shift = [torch.roll(y_c, shift, 2) for y_c, shift in zip(ys, range(-self.pad, self.pad + 1))]
        y_cat = torch.cat(y_shift, 1)
        y_cat = torch.narrow(y_cat, 2, self.pad, H)
        y_s = torch.narrow(y_cat, 3, self.pad, W)

        y_s = y_s.reshape(B, C, H * W).contiguous()
        y_shift_r = y_s.transpose(1, 2)

        y = self.fc2(y_shift_r)
        y = self.drop(y)

        out = torch.cat((x, y), dim=2)
        out = self.dwconv_(out, H, W)
        return out

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
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
    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DWConv(nn.Module):
    def __init__(self, dimin=768, dimout=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dimin, dimout, 3, 1, 1, bias=True, groups=dimout)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# SENet
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# GFNet
class GlobalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
    def forward(self, x):
        B, C, a, b = x.shape
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho')
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# DenseSPF
class DenseSPF(nn.Module):
    def __init__(self, dim, scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)
    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        t = 0
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = t + spx[i]
            sp = self.convs[i](sp)
            t = t + sp
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        return x

class llra(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Linear(dim, dim, bias=bias)
        self.qk = nn.Linear(dim, dim * 2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        qk = self.qk(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# LRRA
class LRRA(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 num_heads=8, bias=True, attn_drop=0., drop=0., qk_scale=None):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.llra = llra(dim, num_heads=num_heads, bias=bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.drop_path(self.gamma * self.llra(self.norm(x), H, W))
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

class MixUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=512,
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        # self.filters = [8, 16, 32, 64, 128]    # S
        self.filters = [16, 32, 128, 160, 256]    # M
        # self.filters = [32, 64, 128, 256, 512]    # L

        self.encoder1 = nn.Conv2d(input_channels, self.filters[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(self.filters[0], self.filters[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(self.filters[1], self.filters[2], 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(self.filters[2], self.filters[3], 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(self.filters[3], self.filters[4], 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(self.filters[0])
        self.ebn2 = nn.BatchNorm2d(self.filters[1])
        self.ebn3 = nn.BatchNorm2d(self.filters[2])
        self.ebn4 = nn.BatchNorm2d(self.filters[3])
        self.ebn5 = nn.BatchNorm2d(self.filters[4])

        self.norm1 = norm_layer(self.filters[0])
        self.norm2 = norm_layer(self.filters[1])
        self.norm3 = norm_layer(self.filters[2])
        self.norm4 = norm_layer(self.filters[3])
        self.norm5 = norm_layer(self.filters[4])

        self.dnorm1 = norm_layer(self.filters[3])
        self.dnorm2 = norm_layer(self.filters[2])
        self.dnorm3 = norm_layer(self.filters[1])
        self.dnorm4 = norm_layer(self.filters[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=self.filters[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=self.filters[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block3 = nn.ModuleList([shiftedBlock(
            dim=self.filters[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block4 = nn.ModuleList([shiftedBlock(
            dim=self.filters[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block5 = nn.ModuleList([shiftedBlock(
            dim=self.filters[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=self.filters[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=self.filters[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock3 = nn.ModuleList([shiftedBlock(
            dim=self.filters[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock4 = nn.ModuleList([shiftedBlock(
            dim=self.filters[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.decoder1 = nn.Conv2d(self.filters[4], self.filters[3], 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(self.filters[3], self.filters[2], 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(self.filters[2], self.filters[1], 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(self.filters[1], self.filters[0], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(self.filters[0], self.filters[0], 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(self.filters[3])
        self.dbn2 = nn.BatchNorm2d(self.filters[2])
        self.dbn3 = nn.BatchNorm2d(self.filters[1])
        self.dbn4 = nn.BatchNorm2d(self.filters[0])

        self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        # DenseSPF
        self.DenseSPF1 = DenseSPF(dim=self.filters[0], scales=2)
        self.DenseSPF2 = DenseSPF(dim=self.filters[1], scales=3)
        self.DenseSPF3 = DenseSPF(dim=self.filters[2], scales=4)
        self.DenseSPF4 = DenseSPF(dim=self.filters[3], scales=5)
        self.DenseSPF5 = DenseSPF(dim=self.filters[4], scales=6)
        # LRRA
        self.LRRA1 = LRRA(dim=self.filters[0], drop_path=0, num_heads=8)
        self.LRRA2 = LRRA(dim=self.filters[1], drop_path=0, num_heads=8)
        self.LRRA3 = LRRA(dim=self.filters[2], drop_path=0, num_heads=8)
        self.LRRA4 = LRRA(dim=self.filters[3], drop_path=0, num_heads=8)
        self.LRRA5 = LRRA(dim=self.filters[4], drop_path=0, num_heads=8)
        # CA
        self.att1 = SELayer(self.filters[0])
        self.att2 = SELayer(self.filters[1])
        self.att3 = SELayer(self.filters[2])
        self.att4 = SELayer(self.filters[3])
        self.att5 = SELayer(self.filters[4])
        # FD
        self.sizes = [img_size // 2, img_size // 4, img_size // 8, img_size // 16, img_size // 32]
        self.fatt4 = GlobalFilter(self.filters[3], h=self.sizes[3], w=(self.sizes[3] // 2 + 1))

    def forward(self, x):
        B = x.shape[0]
        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        # DenseSPF
        out = self.DenseSPF1(out)
        # LRRA
        out = self.LRRA1(out)
        # DSMLP
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # CA
        out = self.att1(out)
        t1 = out

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        out = self.DenseSPF2(out)
        out = self.LRRA2(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att2(out)
        t2 = out

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        out = self.DenseSPF3(out)
        out = self.LRRA3(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att3(out)
        t3 = out

        ### Stage 4
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        out = self.DenseSPF4(out)
        out = self.LRRA4(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block4):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att4(out)
        t4 = out
        t4 = self.fatt4(t4)

        ### Bottleneck(5)
        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        out = self.DenseSPF5(out)
        out = self.LRRA5(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block5):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att5(out)

        ### Stage 4
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        out = self.DenseSPF4(out)
        out = self.LRRA4(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att4(out)

        ### Stage 3
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        out = self.DenseSPF3(out)
        out = self.LRRA3(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att3(out)

        ## Stage 2
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = self.DenseSPF2(out)
        out = self.LRRA2(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att2(out)

        ### Stage 1
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = self.DenseSPF1(out)
        out = self.LRRA1(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att1(out)

        ### Stage 0
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = MixUNet(1).cuda()

    flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)