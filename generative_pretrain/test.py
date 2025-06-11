import numpy as np
import time
import torch
import glob
import argparse
import torch.nn as nn
# from accelerate import Accelerator
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

import sys
sys.path.append('../models')
from network_utils   import *
from dataset         import MyDataset
from Diffusion_Basic import Diffusion_Basic
from UNet_Basic      import UNet_Basic

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
# UNet
parser.add_argument('--img_size',           type=int,  default=64)
parser.add_argument('--self_condition',     type=bool, default=True)
parser.add_argument('--dim_mults',          type=int, nargs='+', default=[1, 2, 4, 8])
# Diffusion
parser.add_argument('--timesteps',          type=int,  default=1000)
parser.add_argument('--sampling_timesteps', type=int,  default=150)
parser.add_argument('--beta_schedule',      type=str,  default='linear')
# Testing
parser.add_argument('--checkpoint',         type=str,  default='./results/model-8.pt')
parser.add_argument('--save_name',          type=str,  default='test_image')
parser.add_argument('--data_folder',        type=str,  default='/cluster/project7/backup_masramon/IQT/PICAI/ADC/')
parser.add_argument('--batch_size',         type=int,  default=5)
parser.add_argument('--is_pretrain',        action='store_true')
parser.add_argument('--finetune',           dest='is_pretrain', action='store_false')
parser.set_defaults(is_pretrain=True)
args, unparsed = parser.parse_known_args()

def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    psnr_val = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim_metric(gt_np, pred_np, data_range=1.0)
    return psnr_val, ssim_val

def evaluate_results(diffusion, dataloader, device):
    psnr_list, ssim_list = [], []
    for batch in dataloader:
        highres   = batch['HighRes'].to(device)
        lowres    = batch['LowRes'].to(device)

        with torch.no_grad():
            pred = diffusion.sample(batch_size=highres.size(0), cond=lowres)
        
        for j in range(pred.size(0)):
            psnr_val, ssim_val = compute_metrics(pred[j], highres[j])
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')


def format_image(tensor):
    # Convert from torch tensor [1, H, W] -> [H, W], ensure it's on CPU and NumPy
    return tensor.squeeze().detach().cpu().numpy()


def visualize_batch(diffusion, dataloader, batch_size, device, output_name="test_image"):
    batch   = next(iter(dataloader))
    highres = batch['HighRes'].to(device)
    lowres  = batch['LowRes'].to(device)

    with torch.no_grad():
        pred = diffusion.sample(batch_size=batch_size, cond=lowres)
    
    fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(3*3,3*batch_size))
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('Super resolution (Output)')
    axes[0,2].set_title('High res (Ground truth)')

    for i in range(batch_size):
        im0 = axes[i, 0].imshow(format_image(lowres[i]),  cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        fig.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].imshow(format_image(pred[i]),    cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1])
        
        im2 = axes[i, 2].imshow(format_image(highres[i]), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        fig.colorbar(im2, ax=axes[i, 2])

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

 
def main():
    # accelerator = Accelerator(split_batches=True, mixed_precision='no')
    # device      = accelerator.device
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet_Basic(
        dim             = args.img_size,
        dim_mults       = tuple(args.dim_mults),
        self_condition  = args.self_condition,
    )

    diffusion = Diffusion_Basic(
        model,
        image_size          = args.img_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule
    )

    print('Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])
    
    # Move model to device
    model.eval()
    model.to(device)
    diffusion.model = model
    diffusion.to(device)
    
    print('Loading data...')
    dataset     = MyDataset(args.data_folder, args.img_size, is_train=False, is_pretrain=args.is_pretrain) 
    dataloader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # print('Evaluating...')
    # evaluate_results(diffusion, dataloader, device)
    
    print('Visualising...')
    visualize_batch(diffusion, dataloader, args.batch_size, device, output_name=args.save_name)
    
    

    
    
    
if __name__ == '__main__':
    main()
