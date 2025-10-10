import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2 

from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error      as mse_metric

import sys
sys.path.append('../models')
from models.network_utils import *

def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    
    mse  = mse_metric (gt_np, pred_np)
    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0)
    return mse, psnr, ssim

def evaluate_results(diffusion, dataloader, device, batch_size, use_T2W=False):
    mse_list, psnr_list, ssim_list = [], [], []
    for batch in dataloader:
        highres   = batch['HighRes'].to(device)
        lowres    = batch['LowRes'].to(device)
        
        if use_T2W:
            try:
                t2w_input = [np.squeeze(i).to(device) for i in batch['T2W_embed']]
                t2w_input = (t2w_input[0].unsqueeze(1), t2w_input[1], t2w_input[2], t2w_input[3])
            except:
                t2w_input = batch['T2W'].to(device)
                
            with torch.no_grad():
                pred = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input)
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


def plot_image(image, fig, axes, i, j, colorbar=True):
    image  = format_image(image)
            
    img_plot = axes[i, j].imshow(image,  cmap='gray', vmin=0, vmax=1)
    axes[i, j].axis('off')
    if colorbar:
        fig.colorbar(img_plot, ax=axes[i, j])
    
def visualize_batch(diffusion, dataloader, batch_size, device, use_T2W=False, output_name="test_image"):
    ncols = 5 if use_T2W else 4
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('Super resolution (Output)')
    axes[0,2].set_title('Error')
    axes[0,3].set_title('High res (Ground truth)')
    if use_T2W:
        axes[0,4].set_title('High res T2W')
        
    # Get images 
    batch   = next(iter(dataloader))
    highres = batch['HighRes'].to(device)
    lowres  = batch['LowRes'].to(device)
    
    if diffusion is not None:
        if use_T2W:
            try:
                t2w_input = [np.squeeze(i).to(device) for i in batch['T2W_embed']]
                t2w_input = (t2w_input[0].unsqueeze(1), t2w_input[1], t2w_input[2], t2w_input[3])
                t2w_image = False
            except:
                t2w_input = batch['T2W'].to(device)
                t2w_image = True

            with torch.no_grad():
                pred  = diffusion.sample(lowres, batch_size=lowres.shape[0], t2w=t2w_input)
        else:
            with torch.no_grad():
                pred  = diffusion.sample(lowres, batch_size=lowres.shape[0])
    else:
        lowres    = batch['LowRes']
        full_size = lowres.shape[-1]
        for i in range(batch_size):
            lowres[i] = cv2.resize(lowres[i], (full_size//2,full_size//2))
            lowres[i] = cv2.resize(lowres[i], (full_size,full_size), interpolation=cv2.INTER_LINEAR)
        lowres    = lowres.to(device)

    for i in range(batch_size):
        # Plot images
        plot_image(lowres[i],  fig, axes, i, 0)
        plot_image(pred[i],    fig, axes, i, 1)
        plot_image(pred[i],    fig, axes, i, 2, False)
        plot_image(highres[i], fig, axes, i, 3)
        if use_T2W:
            if t2w_image:
                plot_image(t2w_input[i], fig, axes, i, 4)
            else:
                plot_image(t2w_input[0][i], fig, axes, i, 4)

        # Error
        err = np.abs(format_image(pred[i]) - format_image(highres[i]))
        p99 = np.percentile(err, 99.5)
        den = p99 if p99 > 1e-8 else (err.max() + 1e-8)
        err_norm = np.clip(err / den, 0, 1)

        im_overlay = axes[i, 2].imshow(err_norm, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.6)
        cbar = fig.colorbar(im_overlay, ax=axes[i, 2])

    fig.tight_layout(pad=0.25)
    save_path = os.path.join('./test_images', output_name+'.jpg')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

 