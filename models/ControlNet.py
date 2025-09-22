import torch
from torch      import nn
from functools  import partial
from network_utils   import *
from network_modules import *  

class ControlNet(nn.Module):
    """
    Produces a list of residual tensors that align with UNet_Basic's injection sites:
      - 2 residuals per encoder level (after each ResBlock)
      - 2 residuals at the mid (after block1 and block2)
      - 2 residuals per decoder level (after each ResBlock)
      - 1 residual at the final ResBlock (optional but handy)
    """
    def __init__(
        self, 
        dim                 = 64, 
        dim_mults           = (1,2,4,8), 
        control_in_channels = 1, 
        with_time_emb       = True
    ):
        super().__init__()
        
        self.dim            = dim
        self.dim_mults      = dim_mults
        self.block_klass    = partial(ResnetBlock, groups=8)

        self.init_conv = nn.Conv2d(control_in_channels, dim, 7, padding=3)
        
        dims_mask        = [dim, *map(lambda m: dim*m, dim_mults)]
        self.in_out_mask = list(zip(dims_mask[:-1], dims_mask[1:]))

        # time embedding (match UNet)
        if with_time_emb:
            self.time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim*4), nn.GELU(),
                nn.Linear(dim*4, dim)
            )
        else:
            self.time_dim = None
            self.time_mlp = None

        self.downs      = nn.ModuleList([])
        self.down_projs = nn.ModuleList([])
        self.ups        = nn.ModuleList([])
        self.up_projs   = nn.ModuleList([])
        
        self.num_resolutions = len(self.in_out_mask)

        # ---- DOWN ---- #
        for ind, (dim_in, dim_out) in enumerate(self.in_out_mask):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                self.block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),

                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
            
            # residuals should match the post-block activations (dim_in channels)
            self.down_projs.append(nn.ModuleList([ZeroConv2d(dim_in, dim_in), ZeroConv2d(dim_in, dim_in)]))
 
        # ---- MID ---- #
        mid_dim = dims_mask[-1]
        
        self.mid_block1 = self.block_klass(mid_dim, mid_dim, time_emb_dim=self.time_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2 = self.block_klass(mid_dim, mid_dim, time_emb_dim=self.time_dim)
        
        self.mid_proj1  = ZeroConv2d(mid_dim, mid_dim)
        self.mid_proj2  = ZeroConv2d(mid_dim, mid_dim)

        # ---- UP ---- #
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out_mask)):
            is_last = ind >= (self.num_resolutions - 1)
            
            self.ups.append(nn.ModuleList([
                # UNet takes in concatenated inputs (dim_in*3 or dim_in*2), but ControlNet injects after the block (dim_in)
                self.block_klass(dim_in, dim_in, time_emb_dim=self.time_dim),
                self.block_klass(dim_in, dim_in, time_emb_dim=self.time_dim),
                
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))

            self.up_projs.append(nn.ModuleList([ZeroConv2d(dim_in, dim_in), ZeroConv2d(dim_in, dim_in)]))

        # ---- FINAL ---- #
        self.final_res_block = self.block_klass(dim, dim, time_emb_dim=self.time_dim)
        self.final_proj      = ZeroConv2d(dim, dim)

    def forward(self, control, time):
        """
        Returns list of residuals in the order they should be added to UNet_Basic's forward.
        """
        x = self.init_conv(control)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        
        residuals = []

        # ---- DOWN ---- #
        for (conv1, conv2, downsample), projs in zip(self.downs, self.down_projs):
            x = conv1(x, t);    residuals.append(projs[0](x))
            x = conv2(x, t);    residuals.append(projs[1](x))
            x = downsample(x)

        # ---- MID ---- #
        x = self.mid_block1(x, t); residuals.append(self.mid_proj1(x))
        x = self.mid_attn(x)
        x = self.mid_block2(x, t); residuals.append(self.mid_proj2(x))

        # ---- UP ---- #
        for (conv1, conv2, upsample), projs in zip(self.ups, self.up_projs):
            x = conv1(x, t);    residuals.append(projs[0](x))
            x = conv2(x, t);    residuals.append(projs[1](x))
            x = upsample(x)

        x = self.final_block(x, t); residuals.append(self.final_proj(x))
        return residuals

