import numpy as np
import time
import torch
import glob
import argparse
import torch.nn as nn
from accelerate import Accelerator
import os
import sys
sys.path.append('../models')
from Diffusion_Basic import Diffusion_Basic
from UNet_Basic      import UNet_Basic
from trainer_class   import Trainer

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
# UNet
parser.add_argument('--img_size',           type=int,  default=64)
parser.add_argument('--self_condition',     type=bool, default=True)
parser.add_argument('--dim_mults',          type=int,  nargs='+', default=[1, 2, 4, 8])
# Diffusion
parser.add_argument('--timesteps',          type=int,  default=1000)
parser.add_argument('--sampling_timesteps', type=int,  default=100)
parser.add_argument('--beta_schedule',      type=str,  default='linear')
parser.add_argument('--perct_λ',            type=float,default=0.1)
# Training
parser.add_argument('--data_folder',        type=str,  default='/cluster/project7/backup_masramon/IQT/PICAI/ADC/')
parser.add_argument('--results_folder',     type=str,  default='./results')
parser.add_argument('--batch_size',         type=int,  default=16)
parser.add_argument('--lr',                 type=float,default=8e-5)
parser.add_argument('--n_epochs',           type=int,  default=10000)
parser.add_argument('--ema_decay',          type=float,default=0.995)
parser.add_argument('--save_every',         type=int,  default=500)
parser.add_argument('--sample_every',       type=int,  default=500)

args, unparsed = parser.parse_known_args()



def main():
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

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
        beta_schedule       = args.beta_schedule,
        perct_λ             = args.perct_λ
    )

    trainer = Trainer(
        diffusion,
        args.data_folder,
        accelerator,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        train_num_steps     = args.n_epochs,
        gradient_accumulate_every = 2,
        ema_decay           = args.ema_decay,
        amp                 = False,
        calculate_fid       = False,
        results_folder      = args.results_folder,
        save_every          = args.save_every ,
        sample_every        = args.sample_every
    )

    trainer.train()
    
if __name__ == '__main__':
    print(args)
    main()
