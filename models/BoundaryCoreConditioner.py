import torch
from torch import nn
import numpy as np
from functools        import partial
from pvt_v2           import pvt_v2_b2
from network_utils    import *
from network_modules  import *

class ConditionExtractor(nn.Module):
    def __init__(
        self,
        dim,
        cp_condition_net,
        dim_mults           = (1, 2, 4, 8),
        self_condition      = True,
        with_time_emb       = True,
        residual            = False,
        input_img_channels  = 1
    ):
        super(ConditionExtractor, self).__init__()

        # determizne dimensions
        mask_channels       = 1
        self.input_img_channels = input_img_channels
        self.mask_channels      = mask_channels
        self.self_condition     = self_condition

        output_channels     = mask_channels
        mask_channels       = mask_channels * (2 if self_condition else 1)
        self.init_conv      = nn.Conv2d(mask_channels, dim, 7, padding = 3)
        self.init_conv_cond = nn.Conv2d(input_img_channels, dim, 7, padding = 3)

        self.channels = self.input_img_channels
        self.residual = residual
        dims_rgb  = [dim, *map(lambda m: dim * m, dim_mults)]
        dims_mask = [dim, *map(lambda m: dim * m, dim_mults)]

        in_out_rgb  = list(zip(dims_rgb[:-1], dims_rgb[1:]))
        in_out_mask = list(zip(dims_mask[:-1], dims_mask[1:]))
        
        block_klass = partial(ResnetBlock, groups = 8)

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

        self.side_out_for_core      = nn.ModuleList([])
        self.side_out_for_boundary  = nn.ModuleList([])

        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out_mask)

        in_out_mask = [(2,64),(64,128),(128,320),(320,512)]

        for ind, (dim_in, dim_out) in enumerate(in_out_mask):
            is_last = ind >= (num_resolutions - 1)
            self.side_out_for_core.append(nn.ModuleList([ ConvBlock(dim_out, 64) ]))
            self.side_out_for_boundary.append(nn.ModuleList([ ConvBlock(dim_out, 64) ]))
        
        backbone = PyramidVisionTransformerV2(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
        backbone.default_cfg = _cfg()
        pretrained_dict   = torch.load(cp_condition_net)
        pretrained_dict   = {k: v for k, v in pretrained_dict.items() if k not in ["patch_embed1.proj.weight"]}
        backbone.load_state_dict(pretrained_dict, strict=False)

        self.backbone           = torch.nn.Sequential(*list(backbone.children()))[:-1]
        self.decoder_core       = CoreEnhance((64, 128, 256, 512), (8, 16, 32, 64), out_c =64)
        self.decoder_boundary   = BoundaryEnhance((64, 128, 256, 512), (8, 16, 32, 64), out_c =64)
        self.core_boundary      = DecoderCoreBoundary(out_c =64)

    def normalization(channels):
        """
        Make a standard normalization layer.

        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid
    # def forward(self, input):
    def forward(self, input):

        B,C,H,W, = input.shape
        cond = input

        pyramid = self.get_pyramid(cond)

        if self.residual:
            orig_x = input

        side_out_body = []
        for i, ModuleList in enumerate(self.side_out_for_core):
            for conv in ModuleList:
                side = conv(pyramid[i])
                side_out_body.append(side)

        side_out_detail = []
        for i, ModuleList in enumerate(self.side_out_for_boundary):
            for conv in ModuleList:
                side = conv(pyramid[i])
                side_out_detail.append(side)

        body_side = [tensor.clone() for tensor in side_out_body]
        # body_side = body_side[::-1]

        detail_side = [tensor.clone() for tensor in side_out_detail]
        # detail_side = detail_side[::-1]

        _, out_body = self.decoder_core(body_side)
        _, out_detail = self.decoder_boundary(detail_side)

        out_body_detail = self.core_boundary(out_body, out_detail)
        return out_body, out_detail, out_body_detail, pyramid[-1]


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    model = ConditionExtractor(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        residual=True
    ).cuda()
    input_R = torch.randn(1,1,256,256).cuda()
    label_noise_t = torch.randn(1,1,256,256).cuda()
    time = torch.randn(2).cuda()
    X=model(input_R)