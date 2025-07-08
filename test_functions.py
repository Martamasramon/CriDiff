import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error      as mse_metric

import sys
sys.path.append('../models')
from network_utils   import *

def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    
    mse  = mse_metric (gt_np, pred_np)
    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0)
    return mse, psnr, ssim

def evaluate_results(diffusion, dataloader, device, batch_size, t2w=False):
    mse_list, psnr_list, ssim_list = [], [], []
    for batch in dataloader:
        highres   = batch['HighRes'].to(device)
        lowres    = batch['LowRes'].to(device)
        if t2w:
            t2w_img = [np.squeeze(i).to(device) for i in batch['T2W']]
            with torch.no_grad():
                pred = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_img)
        else:
            with torch.no_grad():
                pred = diffusion.sample(lowres, batch_size=lowres.shape[0])
        
        for j in range(pred.size(0)):
            mse, psnr, ssim = compute_metrics(pred[j], highres[j])
            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    print(f'Average MSE:  {np.mean(mse_list):.6f}')
    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')


def format_image(tensor):
    # Convert from torch tensor [1, H, W] -> [H, W], ensure it's on CPU and numpy
    return tensor.squeeze().detach().cpu().numpy()


def visualize_batch(diffusion, dataloader, batch_size, device, t2w=False, output_name="test_image"):
    ncols = 4 if t2w else 3
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('Super resolution (Output)')
    axes[0,2].set_title('High res (Ground truth)')
    if t2w:
        axes[0,3].set_title('High res T2W')
        
    # Get images 
    batch   = next(iter(dataloader))
    highres = batch['HighRes'].to(device)
    lowres  = batch['LowRes'].to(device)
    if t2w:
        t2w_input = [np.squeeze(i).to(device) for i in batch['T2W']]
        t2w_img   = batch['T2W_img'].to(device)
        with torch.no_grad():
            pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input)
    else:
        with torch.no_grad():
            pred  = diffusion.sample(lowres, batch_size=lowres.shape[0])

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
        
        if t2w:
            im3 = axes[i, 3].imshow(format_image(t2w_img[i]), cmap='gray', vmin=0, vmax=1)
            axes[i, 3].axis('off')
            fig.colorbar(im3, ax=axes[i, 3])

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

 