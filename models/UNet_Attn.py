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
    def __init__(self, *args, cp_condition_net=None, **kwargs):
        super().__init__(*args, **kwargs)

        ### New layers 
        self.condition_extractor = ConditionExtractor(self.dim, cp_condition_net, self.dim_mults, False, False, residual)
        self.cond_mid_block1     = copy.deepcopy(self.mid_block1) 

        for i in range(self.num_resolutions):
            self.body_sup.append(nn.ModuleList([
                nn.Conv2d(self.side_unit_channel, 1, kernel_size=3, stride=1, padding=1)
            ]))
            self.detail_sup.append(nn.ModuleList([
                nn.Conv2d(self.side_unit_channel, 1, kernel_size=3, stride=1, padding=1)
            ]))
            self.side_out_sup.append(nn.ModuleList([
                nn.Conv2d(self.side_unit_channel, 1, kernel_size=3, stride=1, padding=1)
            ]))
            

        ### Redefine these layers
        self.downs_input        = nn.ModuleList([])
        self.downs_label_noise  = nn.ModuleList([])
        self.ups                = nn.ModuleList([])
        
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(self.in_out_mask, self.full_self_attn)):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.downs_label_noise.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=self.side_unit_channel))), ## <---- not in basic
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(self.in_out_mask), reversed(self.full_self_attn))):

            is_last = ind >= (self.num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_in*3, dim_in, time_emb_dim = time_dim) if ind < 3 else block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearCrossAttention(dim_in, context_in=self.side_unit_channel))), ## <---- not in basic
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))


    def forward(self, input_x, label_noise_t, time, x_self_cond):
        B,C,H,W, = input_x.shape
        x = label_noise_t
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(input_x))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.residual:
            orig_x = input_x

        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        out_body, out_detail, out_body_detail, cond= self.condition_extractor(input_x)

        side_for_sup = []
        body_detail_sup = [tensor.clone() for tensor in out_body_detail]
        body_detail_sup = body_detail_sup[::-1]

        body_for_sup = []
        body_sup = [tensor.clone() for tensor in out_body]
        body_sup = body_sup[::-1]

        detail_for_sup = []
        detail_sup = [tensor.clone() for tensor in out_detail]
        detail_sup = detail_sup[::-1]


        for i, ModuleList in enumerate(self.side_out_sup):
            for conv in ModuleList:
                side = conv(body_detail_sup[i])
                side = F.interpolate(side, size=[H, W], mode='bilinear')
                side_for_sup.append(side)

        for i, ModuleList in enumerate(self.body_sup):
            for conv in ModuleList:
                side = conv(body_sup[i])
                side = F.interpolate(side, size=[H, W], mode='bilinear')
                body_for_sup.append(side)

        for i, ModuleList in enumerate(self.detail_sup):
            for conv in ModuleList:
                side = conv(detail_sup[i])
                side = F.interpolate(side, size=[H, W], mode='bilinear')
                detail_for_sup.append(side)

        # label_noise
        out_body_detail_for_label_decoder   = [tensor.clone() for tensor in out_body_detail]
        out_body_for_label_decoder          = [tensor.clone() for tensor in out_body]
        out_detail_for_label_decoder        = [tensor.clone() for tensor in out_detail]

        label_noise_side = []
        num = 0
        out_body_detail = out_body_detail[::-1]
        out_body        = out_body[::-1]
        out_detail      = out_detail[::-1]
        
        for convnext, convnext2, LinearCorssAttention, downsample in self.downs_label_noise:
            if num == 0:
                if x.shape != out_body_detail[0].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[0], size=(H,W), mode="bilinear")
                x = x + condition

            if num == 1:
                if x.shape != out_body_detail[1].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[1], size=(H,W), mode="bilinear")
                x = x + condition

            if num == 2:
                if x.shape != out_body_detail[2].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[2], size=(H,W), mode="bilinear")

                x = x + torch.cat((condition, condition), dim=1)

            if num == 3:
                if x.shape != out_body_detail[3].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[3], size=(H,W), mode="bilinear")
                x = x + torch.cat((condition, condition, condition, condition), dim=1)


            x = convnext(x, t)
            label_noise_side.append(x)

            x = convnext2(x, t)
            if num <= 1:
                context = out_body[num]
            if num > 1:
                context = out_detail[num]
            x = LinearCorssAttention(x, context)
            label_noise_side.append(x)
            x = downsample(x)
            num = num + 1

        x = self.mid_block1(x, t)
        cond = self.cond_mid_block1(cond, t)
        if x.shape != cond.shape:
            B, C, H, W, = x.shape
            cond = F.interpolate(cond, size=(H, W), mode="bilinear")
        x = x + cond
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        num = 0
        for convnext, convnext2, LinearCorssAttention, upsample in self.ups:
            if num == 0:
                if x.shape != out_body_detail_for_label_decoder[0].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[0], size=(H,W), mode="bilinear")

                condition = torch.cat((condition, condition, condition, condition), dim=1)
            if num == 1:
                if x.shape != out_body_detail_for_label_decoder[1].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[1], size=(H,W), mode="bilinear")

                condition = torch.cat((condition, condition), dim=1)
            if num == 2:
                if x.shape != out_body_detail_for_label_decoder[2].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[2], size=(H,W), mode="bilinear")
                condition = condition
            if num == 3:
                if x.shape != out_body_detail_for_label_decoder[3].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[3], size=(H,W), mode="bilinear")
                condition = condition
            
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = x + condition
            x = torch.cat((x, label_noise_side.pop()), dim=1)

            x = convnext2(x, t)
            if num <= 1:
                x = LinearCorssAttention(x, out_detail_for_label_decoder[num])
            if num > 1:
                x = LinearCorssAttention(x, out_body_for_label_decoder[num])
            x = upsample(x)
            num = num + 1

        if self.residual:
            return self.final_conv(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x), side_for_sup, body_for_sup, detail_for_sup


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    model = UNet_Attn(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        residual=False
    ).cuda()

    input_R = torch.randn(1,1,256,256).cuda()
    label_noise_t = torch.randn(1,1,256,256).cuda()
    time = torch.randn(2).cuda()
    X=model(input_R, label_noise_t, time, None)