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
        residual        = False
    ):
        super().__init__()

        # determine dimensions
        self.dim                = dim
        self.dim_mults          = dim_mults
        self.self_condition     = self_condition
        self.with_time_emb      = with_time_emb
        self.residual           = residual 
        
        self.input_img_channels = 1
        self.mask_channels      = 1
        self.side_unit_channel  = 64

        output_channels = self.mask_channels
        mask_channels   = self.mask_channels * (2 if self_condition else 1)
        self.init_conv  = nn.Conv2d(mask_channels, dim, 7, padding = 3)

        self.channels = self.input_img_channels
        self.residual = residual
        dims_mask     = [dim, *map(lambda m: dim * m, dim_mults)]

        self.in_out_mask = list(zip(dims_mask[:-1], dims_mask[1:]))
        block_klass = partial(ResnetBlock, groups = 8)

        self.full_self_attn: tuple = (False, False, False, True)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs_input        = nn.ModuleList([])
        self.downs_label_noise  = nn.ModuleList([])
        self.side_out_sup       = nn.ModuleList([])
        self.body_sup           = nn.ModuleList([])
        self.detail_sup         = nn.ModuleList([])
        self.ups                = nn.ModuleList([])

        self.num_resolutions = len(in_out_mask)

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(self.in_out_mask, self.full_self_attn)):
            is_last = ind >= (self.num_resolutions - 1)

            self.downs_label_noise.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),

                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims_mask[-1]
        
        self.mid_block1     = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn       = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2     = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)


        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(self.in_out_mask), reversed(self.full_self_attn))):

            is_last = ind >= (self.num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_in*3, dim_in, time_emb_dim = time_dim) if ind < 3 else block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))

        self.final_res_block = block_klass(dim, dim, time_emb_dim = time_dim)
        self.final_conv      = nn.Sequential(
            nn.Conv2d(dim, output_channels, 1),
        )


    def normalization(channels):
        """
        Make a standard normalization layer.

        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)


    def forward(self, input_x, time, x_self_cond):     
        B,C,H,W, = input_x.shape
        device   = input_x.device
        x = input_x
        
        if self.self_condition:
            tens = torch.zeros((B,1,H,W)).to(device)
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(tens))
            x = torch.cat((x, x_self_cond), dim=1)
        if self.residual:
            orig_x = input_x

        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        label_noise_side = []
        num = 0
        
        try:
            for convnext, convnext2, downsample in self.downs_label_noise:
                x = convnext(x, t)
                label_noise_side.append(x)
                x = convnext2(x, t)
                label_noise_side.append(x)
                x = downsample(x)
                num = num + 1
        except:
            for convnext, convnext2 in self.downs_label_noise:
                x = convnext(x, t)
                label_noise_side.append(x)
                x = convnext2(x, t)
                label_noise_side.append(x)
                num = num + 1

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        num = 0
        for convnext, convnext2, upsample in self.ups:

            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = torch.cat((x, label_noise_side.pop()), dim=1)

            x = convnext2(x, t)
            x = upsample(x)
            num = num + 1

        if self.residual:
            return self.final_conv(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    model = UNet_Basic(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        residual=False
    ).cuda()
    input_R = torch.randn(1,1,256,256).cuda()
    label_noise_t = torch.randn(1,1,256,256).cuda()
    time = torch.randn(2).cuda()
    X=model(input_R, time, None)