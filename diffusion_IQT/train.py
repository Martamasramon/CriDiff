import numpy as np
import os
import time
import torch
torch.cuda.set_device(0)
import glob
import argparse
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from Diffusion_Attn import Diffusion_Attn
from UNet_Attn      import UNet_Attn

import sys
sys.path.append('../')
from trainer_class  import Trainer
from dataset        import MyDataset

folder = '/cluster/project7/backup_masramon/IQT/'

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
# UNet
parser.add_argument('--img_size',           type=int,  default=64)
parser.add_argument('--self_condition',     type=bool, default=True)
parser.add_argument('--dim_mults',          type=int,  nargs='+', default=[1, 2, 4, 8])
# Diffusion
parser.add_argument('--timesteps',          type=int,  default=1000)
parser.add_argument('--sampling_timesteps', type=int,  default=150)
parser.add_argument('--beta_schedule',      type=str,  default='linear')
parser.add_argument('--perct_λ',            type=float,default=0.1)
# Training
parser.add_argument('--data_folder',        type=str,  default='PICAI')
parser.add_argument('--finetune',           type=str,  default=None)
parser.add_argument('--surgical_only',      action='store_true')
parser.set_defaults(surgical_only = False)
parser.add_argument('--results_folder',     type=str,  default='./results')
parser.add_argument('--batch_size',         type=int,  default=16)
parser.add_argument('--lr',                 type=float,default=8e-5)
parser.add_argument('--n_epochs',           type=int,  default=40000)
parser.add_argument('--ema_decay',          type=float,default=0.995)
# IQT
parser.add_argument('--use_T2W',            action='store_true')
parser.add_argument('--use_histo',          action='store_true')
parser.add_argument('--use_mask',           action='store_true')
parser.set_defaults(use_T2W   = False)
parser.set_defaults(use_histo = False)
parser.set_defaults(use_mask  = False)
# Log process
parser.add_argument('--save_every',         type=int,  default=1000)
parser.add_argument('--sample_every',       type=int,  default=1000)

args, unparsed = parser.parse_known_args()

def main():
    assert args.use_T2W or args.use_histo == True
    accelerator = Accelerator(split_batches=True, mixed_precision='no')

    model = UNet_Attn(
        dim             = args.img_size,
        dim_mults       = tuple(args.dim_mults),
        self_condition  = args.self_condition,
        use_T2W         = args.use_T2W,
        use_histo       = args.use_histo,
    )
    
    diffusion = Diffusion_Attn(
        model,
        image_size          = args.img_size,
        timesteps           = args.timesteps,
        sampling_timesteps  = args.sampling_timesteps,
        beta_schedule       = args.beta_schedule,
        perct_λ             = args.perct_λ
    )
    
    if args.finetune:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        diffusion.load_state_dict(checkpoint['model'], strict=False)
                
    # Dataset and dataloader   
    train_dataset = MyDataset(
        folder + args.data_folder, 
        data_type       = 'train', 
        image_size      = args.img_size, 
        is_finetune     = args.finetune, 
        surgical_only   = args.surgical_only, 
        use_mask        = args.use_mask,
        use_T2W         = args.use_T2W, 
        use_histo       = args.use_histo,
    ) 
    test_dataset = MyDataset(
        folder + args.data_folder, 
        data_type       = 'test', 
        image_size      = args.img_size, 
        is_finetune     = args.finetune, 
        surgical_only   = args.surgical_only, 
        use_mask        = args.use_mask,
        use_T2W         = args.use_T2W, 
        use_histo       = args.use_histo,
    ) 
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size = args.batch_size, shuffle=False)
        
    trainer = Trainer(
        diffusion,
        train_dataloader,
        test_dataloader,
        accelerator,
        use_T2W             = args.use_T2W,
        use_histo           = args.use_histo,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        train_num_steps     = args.n_epochs,
        gradient_accumulate_every = 2,
        ema_decay           = args.ema_decay,
        amp                 = False,
        results_folder      = args.results_folder,
        save_every          = args.save_every ,
        sample_every        = args.sample_every,
        save_best_and_latest_only = True
    )

    trainer.train()
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
