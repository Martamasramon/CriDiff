import numpy as np
import time
import torch
import glob
import argparse
import torch.nn as nn
from accelerate import Accelerator
import os
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
from UNet_Basic import UNet_Basic
    
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
parser.add_argument('--self_condition',     type=bool, default=True)
parser.add_argument('--batch_size',         type=int,  default=16, help='batch size')
parser.add_argument('--train_num_steps',    type=int,  default=40000, help='num of training epochs')
parser.add_argument('--num_timesteps',      type=int,  default=500)
parser.add_argument('--size',               type=int,  default=128)
parser.add_argument('--dataset_root',       type=str,  default='/home/david/datasets/ProstateSeg/NCI-ISBI/images/train')
parser.add_argument('--job_name',           type=str,  default='experiments_name', help='note for this run')

args, unparsed = parser.parse_known_args()

def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    model = UNet_Basic(
        dim             = 64,
        dim_mults       = (1, 2, 4, 8),
        self_condition  = args.self_condition,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size          = args.size,
        timesteps           = args.num_timesteps,
        sampling_timesteps  = 150,
        beta_schedule       = 'linear'
    )

    trainer = Trainer(
        diffusion,
        args.dataset_root,
        accelerator,
        train_batch_size=args.batch_size,
        train_lr=8e-5,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        calculate_fid=False
    )

    trainer.train()
    
if __name__ == '__main__':
    main()
