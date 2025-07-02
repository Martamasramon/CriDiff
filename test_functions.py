import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

import sys
sys.path.append('../models')
from network_utils   import *

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
            pred = diffusion.sample(lowres, batch_size=batch_size)
        
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
        pred = diffusion.sample(lowres, batch_size=batch_size)
    
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

 