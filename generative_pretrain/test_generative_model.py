import numpy as np
import time
import torch
import glob
import argparse
import torch.nn as nn
# from accelerate import Accelerator
import os
from denoising_diffusion_pytorch import GaussianDiffusion, Dataset
from torch.utils.data            import DataLoader
from UNet_Basic        import UNet_Basic
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

import sys
sys.path.append('../module')
from network_utils   import *

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
parser.add_argument('--input_folder',       type=str,  default='../Datasets/ProstateX/images/test/')
parser.add_argument('--output_folder',      type=str,  default='../Datasets/ProstateX/outputs/')
parser.add_argument('--checkpoint',         type=str,  default='../checkpoints/model-8.pt')
parser.add_argument('--batch_size',         type=int,  default=16,      help='batch size')
parser.add_argument('--train_num_steps',    type=int,  default=40000,   help='num of training epochs')
parser.add_argument('--num_timesteps',      type=int,  default=1000)
parser.add_argument('--size',               type=int,  default=128)
parser.add_argument('--dataset_root',       type=str,  default='/home/david/datasets/ProstateSeg/NCI-ISBI/images/train')
parser.add_argument('--job_name',           type=str,  default='experiments_name', help='note for this run')

args, unparsed = parser.parse_known_args()

def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    psnr_val = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim_metric(gt_np, pred_np, data_range=1.0)
    return psnr_val, ssim_val

def main():
    # accelerator = Accelerator(split_batches=True, mixed_precision='no')
    # device      = accelerator.device
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Creating UNet model...')
    model = UNet_Basic(
        dim=64,
        dim_mults=(1, 2, 4, 8),
    )

    print('Creating diffusion model...')
    diffusion = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=args.num_timesteps,
        sampling_timesteps=150,
        beta_schedule='linear'
    )

    print('Loading checkpoints...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diffusion.load_state_dict(checkpoint['model'], strict=False)
    
    # Move model to device
    model.eval()
    model.to(device)
    diffusion.model = model
    diffusion.to(device)
    
    print('Loading data...')
    dataset = Dataset(args.input_folder, args.size, type='test')
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Inference
    psnr_list = []
    ssim_list = []

    for i, (x_input, file_name) in enumerate(loader):
        print(x_input.shape, file_name)

        x_input      = x_input.to(device)
        x_input_norm = normalize_to_neg_one_to_one(x_input)  # normalize [-1, 1]
        
        # Fake timestep + no self-conditioning
        t = torch.full((x_input.shape[0],), diffusion.num_timesteps - 1, device=device, dtype=torch.long)
                
        with torch.no_grad():
            x_t     = diffusion.q_sample(x_input_norm, t)
            x_start = None
            for t_step in reversed(range(t[0] + 1)):
                x_t, x_start = diffusion.p_sample(x_t, t_step, x_start)
                
        # Convert back to [0, 1]
        pred = unnormalize_to_zero_to_one(x_start)

        for j in range(pred.shape[0]):
            save_image(pred[j], os.path.join(args.output_folder, f'output_{file_name[j]}'))

            psnr_val, ssim_val = compute_metrics(pred[j],x_input[j])
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')

    
    
    
if __name__ == '__main__':
    main()
