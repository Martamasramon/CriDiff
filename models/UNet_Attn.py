# Realtive import
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import copy

import os
from network_utils   import *
from network_modules import *
from functools import partial

from UNet_Basic import UNet_Basic


class UNet_Attn(UNet_Basic):
    def __init__(self, *args, cp_condition_net=None, use_T2W=None, use_histo=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_T2W   = use_T2W
        self.use_histo = use_histo

        ### Redefine these layers
        self.downs_label_noise  = nn.ModuleList([])
        self.ups                = nn.ModuleList([])
        
        # Figure out size of context vectors for cross attention
        t2w_channels   = [1, 64, 128, 256]
        histo_channel  = 768
        context_channels = []
        for i in range(len(self.in_out_mask)):
            channels = 0
            if self.use_T2W:
                channels += t2w_channels[i]
            if self.use_histo:
                channels += histo_channel
            context_channels.append(channels)

        for ind, (dim_in, dim_out) in enumerate(self.in_out_mask):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.downs_label_noise.append(nn.ModuleList([
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=context_channels[ind]))), ## <---- not in basic
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out_mask)):
            is_last = ind >= (self.num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                self.block_klass(dim_in*3, dim_in, time_emb_dim = self.time_dim) if ind < 3 else self.block_klass(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in*2, dim_in, time_emb_dim = self.time_dim),
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=context_channels[::-1][ind]))), ## <---- not in basic
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))


    def forward(self, input_x, low_res, time, t2w=None, histo=None, x_self_cond=None):   
        assert not (self.use_T2W   and t2w   is None), "T2W embedding required but not provided"
        assert not (self.use_histo and histo is None), "Histology embedding required but not provided"
        
        ### Make sure histo is size (B, 768, 1, 1). If (B,1) -> histo.view(B, 768, 1, 1)
          
        B,C,H,W, = input_x.shape
        device   = input_x.device
        
        # Concatenate input & lowres image
        x = torch.cat((input_x, low_res), dim=1)
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(input_x))
            x = torch.cat((x_self_cond, x), dim=1)
            
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        label_noise_side = []
        for i, (convnext, convnext2, cross_attention, downsample) in enumerate(self.downs_label_noise):
            x = convnext(x, t)
            label_noise_side.append(x)
            x = convnext2(x, t)
            
            if self.use_T2W and self.use_histo:
                context     = torch.cat((t2w[i], F.interpolate(histo, size=x.shape[2:], mode="bilinear")), dim=1)
            elif self.use_T2W:
                context     = t2w[i]
            elif self.use_histo:
                context     = F.interpolate(histo, size=x.shape[2:], mode="bilinear")
    
            x = cross_attention(x, context)
            label_noise_side.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for i, (convnext, convnext2, cross_attention, upsample) in enumerate(self.ups):
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext2(x, t)
            
            if self.use_T2W and self.use_histo:
                context     = torch.cat((t2w[-(i+1)], F.interpolate(histo, size=x.shape[2:], mode="bilinear")), dim=1)
            elif self.use_T2W:
                context     = t2w[-(i+1)]
            elif self.use_histo:
                context     = F.interpolate(histo, size=x.shape[2:], mode="bilinear") 
            
            x = cross_attention(x, context)
            x = upsample(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x) 

