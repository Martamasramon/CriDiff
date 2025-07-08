import torch
from torch import nn
import numpy as np

import os
from network_utils   import *
from network_modules import *  
from functools import partial

class UNet_Basic(nn.Module):
    def __init__(
        self,
        dim             = 64,
        dim_mults       = (1, 2, 4, 8),
        self_condition  = True,
        with_time_emb   = True,
    ):
        super().__init__()

        # determine dimensions
        self.dim                = dim
        self.dim_mults          = dim_mults
        self.self_condition     = self_condition
        self.with_time_emb      = with_time_emb
        
        self.input_img_channels = 1
        self.mask_channels      = 1

        cond_channels       = 1  # conditioning image
        self_cond_channels  = 1 if self_condition else 0
        input_channels      = self.mask_channels + cond_channels + self_cond_channels
        self.init_conv      = nn.Conv2d(input_channels, dim, 7, padding = 3)

        self.channels = self.input_img_channels
        dims_mask     = [dim, *map(lambda m: dim * m, dim_mults)]

        self.in_out_mask = list(zip(dims_mask[:-1], dims_mask[1:]))
        self.block_klass = partial(ResnetBlock, groups = 8)

        if with_time_emb:
            self.time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.time_dim = None
            self.time_mlp = None

        self.downs_input        = nn.ModuleList([])
        self.downs_label_noise  = nn.ModuleList([])
        self.side_out_sup       = nn.ModuleList([])
        self.body_sup           = nn.ModuleList([])
        self.detail_sup         = nn.ModuleList([])
        self.ups                = nn.ModuleList([])

        self.num_resolutions = len(self.in_out_mask)

        for ind, (dim_in, dim_out) in enumerate(self.in_out_mask):
            is_last = ind >= (self.num_resolutions - 1)

            self.downs_label_noise.append(nn.ModuleList([
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),

                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims_mask[-1]
        
        self.mid_block1     = self.block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn       = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2     = self.block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out_mask)):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.ups.append(nn.ModuleList([
                self.block_klass(dim_in*3, dim_in, time_emb_dim = self.time_dim) if ind < 3 else self.block_klass(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))

        self.final_res_block = self.block_klass(dim, dim, time_emb_dim = self.time_dim)
        self.final_conv      = nn.Sequential(
            nn.Conv2d(dim, self.mask_channels, 1),
        )


    def normalization(channels):
        """
        Make a standard normalization layer.

        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)


    def forward(self, input_x, low_res, time, x_self_cond=None):     
        B,C,H,W, = input_x.shape
        device   = input_x.device
        if input_x.shape!= low_res.shape:
            print(input_x.shape, low_res.shape)
        x = torch.cat((input_x, low_res), dim=1)
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(input_x))
            x = torch.cat((x, x_self_cond), dim=1)
        
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        label_noise_side = []        
        for convnext, convnext2, downsample in self.downs_label_noise:
            x = convnext(x, t)
            label_noise_side.append(x)
            x = convnext2(x, t)
            label_noise_side.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, upsample in self.ups:
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext2(x, t)
            x = upsample(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

