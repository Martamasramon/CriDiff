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
        
        dims = [dim, *map(lambda m: dim*m, dim_mults)]

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
        
        # ---- DOWN ---- #
        num_res = len(dims) - 1
        ch      = dims[0]

        for i in range(num_res):
            next_ch = dims[i+1]
            
            self.downs.append(nn.ModuleList([
                self.block_klass(ch, ch, time_emb_dim = self.time_dim),
                self.block_klass(ch, ch, time_emb_dim = self.time_dim),
                Downsample(ch, next_ch) if i < (num_res - 1) else nn.Conv2d(ch, next_ch, 3, padding = 1)
            ]))
            
            self.down_projs.append(nn.ModuleList([ZeroConv2d(ch, ch), ZeroConv2d(ch, ch)]))
            ch = next_ch 
 
        # ---- MID ---- #
        mid_dim = ch
        
        self.mid_block1 = self.block_klass(mid_dim, mid_dim, time_emb_dim=self.time_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, LinearCrossAttention(mid_dim)))
        self.mid_block2 = self.block_klass(mid_dim, mid_dim, time_emb_dim=self.time_dim)
        
        self.mid_proj1  = ZeroConv2d(mid_dim, mid_dim)
        self.mid_proj2  = ZeroConv2d(mid_dim, mid_dim)

        # ---- UP ---- #
        for i in reversed(range(num_res)):
            prev_ch = dims[i]
            
            self.ups.append(nn.ModuleList([
                # UNet takes in concatenated inputs (dim_in*3 or dim_in*2), but ControlNet injects after the block (dim_in)
                self.block_klass(ch, prev_ch, time_emb_dim=self.time_dim),
                self.block_klass(prev_ch, prev_ch, time_emb_dim=self.time_dim),
                Upsample(prev_ch, prev_ch) if i > 0 else nn.Conv2d(prev_ch, prev_ch, 3, padding=1)
            ]))

            self.up_projs.append(nn.ModuleList([ZeroConv2d(prev_ch, prev_ch), ZeroConv2d(prev_ch, prev_ch)]))
            ch = prev_ch

        # ---- FINAL ---- #
        self.final_block = self.block_klass(ch, ch, time_emb_dim=self.time_dim)
        self.final_proj  = ZeroConv2d(ch, ch)


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

