# Realtive import
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import copy

import os
from network_utils   import *
from network_modules import *
from BoundaryCoreConditioner import ConditionExtractor
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
        
        for ind, (dim_in, dim_out) in enumerate(self.in_out_mask):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.downs_label_noise.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=self.side_unit_channel))), ## <---- not in basic
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out_mask)):
            is_last = ind >= (self.num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_in*3, dim_in, time_emb_dim = time_dim) if ind < 3 else block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=self.side_unit_channel))), ## <---- not in basic
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))


    def forward(self, input_x, time, x_self_cond=None, cond=None, t2w=None, histo=None):     
        B,C,H,W, = input_x.shape
        device   = input_x.device
        x = input_x
        
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(input_x))
            x = torch.cat((x_self_cond, x), dim=1)
            
        if self.residual:
            orig_x = input_x

        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        
        label_noise_side = []
        for i, convnext, convnext2, LinearCrossAttention, downsample in enumerate(self.downs_label_noise):
            x = convnext(x, t)
            label_noise_side.append(x)
            x = convnext2(x, t)
            
            if self.use_T2W:
                x = LinearCrossAttention(x, t2w[i])    
            if self.use_histo:
                histo_level = F.interpolate(histo, size=x.shape[2:], mode="bilinear")
                x = LinearCrossAttention(x, histo_level)    
            
            label_noise_side.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for i, convnext, convnext2, LinearCrossAttention, upsample in enumerate(self.ups):
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = x + condition
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext2(x, t)
            
            if self.use_T2W:
                x = LinearCrossAttention(x, t2w[len(t2w)-i])    
            if self.use_histo:
                histo_level = F.interpolate(histo, size=x.shape[2:], mode="bilinear")
                x = LinearCrossAttention(x, histo_level)    
                
            x = upsample(x)

        if self.residual:
            return self.final_conv(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x) 

