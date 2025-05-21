import torch
import os
from generative_pretrain.denoising_diffusion_pytorch import GaussianDiffusion
from generative_pretrain.UNet_Basic import UNet_Basic  
import argparse
from pathlib import Path
from module.DiffusionModel import DiffSOD
from module.UNet_Attn      import UNet_Attn

def check_checkpoint(checkpoint):
    # If it's a dict, check keys
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", checkpoint.keys())

        if 'model_state_dict' in checkpoint:
            print("-> This looks like a training checkpoint with model and optimizer states.")
        elif 'state_dict' in checkpoint:
            print("-> This might be a model's state_dict wrapped in another dict.")
        elif all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
            print("-> This is probably just a raw state_dict.")
        else:
            print("-> Custom format, check the contents of each key.")
    else:
        print("Checkpoint is not a dict — possibly a full model object.")
        

def dummy_UNet_Basic():
    """Test Basic UNet"""
    # === Settings ===
    save_path = "./checkpoints/dummy_model_basic.pt"
    image_size      = 128
    num_timesteps   = 500
    self_condition  = True

    # === 1. Create UNet model ===
    model = UNet_Basic(
        dim             = 64,
        dim_mults       = (1, 2, 4, 8),
        self_condition  = self_condition,
    )

    # === 2. Wrap in Gaussian Diffusion ===
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=num_timesteps,
        sampling_timesteps=150,
        beta_schedule='linear',
    )

    # === 3. Save model checkpoint ===
    torch.save(model.state_dict(), save_path)
    print(f"Saved dummy checkpoint to {save_path}")
    
    
def dummy_UNet_Attn():
    """Test UNet with Attention"""
    # === Settings ===
    save_path = "./checkpoints/dummy_model_attn.pt"
    cp_condition_net = './checkpoints/pvt_v2_b2.pth' # <---- e.g.
    image_size      = 128
    num_timesteps   = 500
    self_condition  = True

    # === 1. Create UNet model ===
    model = UNet_Attn(
        dim                 = 64,
        dim_mults           = (1, 2, 4, 8),
        self_condition      = self_condition,
        cp_condition_net    = cp_condition_net,
    )
    
    # === 2. Wrap in Gaussian Diffusion ===
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=num_timesteps,
        sampling_timesteps=150,
        beta_schedule='linear',
    )

    # === 3. Save model checkpoint ===
    torch.save(model.state_dict(), save_path)
    print(f"Saved dummy checkpoint to {save_path}")


def compare_keys(pvt_ckpt, unet_ckpt, model_sd):
    def summarize_keys(title, keys):
        print(f"\n🔹 {title} ({len(keys)} keys)")
        for k in list(keys):#[:5]
            print(f"  • {k}")
        # if len(keys) > 5:
        #     print("  ...")

    # summarize_keys("PVT Checkpoint Keys", pvt_ckpt.keys())
    summarize_keys("UNet Checkpoint Keys", unet_ckpt.keys())
    # summarize_keys("DiffSOD State Dict Keys", model_sd.keys())
    
    # ==== Optionally find differences ====
    # print("\nKey Mismatches:")
    # print("- In UNet AND DiffSOD:")
    # print(set(unet_ckpt.keys()) & set(model_sd.keys()))
    # print("- In PVT AND DiffSOD:")
    # print(set(pvt_ckpt.keys()) & set(model_sd.keys()))



def main():
    # ==== Define dummy args for DiffSOD ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_condition_net',   type=str, default='./checkpoints/pvt_v2_b2.pth')
    parser.add_argument('--cp_stage1',          type=str, default='./checkpoints/dummy_model.pt')
    parser.add_argument('--self_condition',     type=bool,default=True)
    parser.add_argument('--num_timesteps',      type=int, default=500)
    parser.add_argument('--beta_sched',         type=str, default='linear')
    parser.add_argument('--size',               type=int, default=128)
    parser.add_argument('--sampling_timesteps', type=int, default=150)
    args = parser.parse_args([])  
    
    # ==== Load Checkpoints ====
    pvt_ckpt  = torch.load(args.cp_condition_net, map_location='cpu')
    unet_ckpt = torch.load(args.cp_stage1, map_location='cpu')['model']
    
    # ==== Init DiffSOD ====
    model = DiffSOD(args, sampling_timesteps=args.sampling_timesteps)
    model_sd = model.state_dict()
    
    print('\n\n')
    compare_keys(pvt_ckpt, unet_ckpt, model_sd)
    

if __name__ == '__main__':
    main()
