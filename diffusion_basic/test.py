import time
import torch
import argparse
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from network_utils   import *
from test_functions  import *
from Diffusion_Basic import Diffusion_Basic
from UNet_Basic      import UNet_Basic

import sys
sys.path.append('../')
from dataset         import MyDataset

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
# General setup
parser.set_defaults(use_histo=False)
parser.set_defaults(use_T2W  =False)
parser.add_argument('--use_histo',          action='store_true')
parser.add_argument('--use_T2W',            action='store_true')
parser.add_argument('--t2w_checkpoint',     type=int,  default = '/cluster/project7/ProsRegNet_CellCount/UNet/checkpoints/checkpoints_0306_1947_stage_1_best.pth')
# UNet
parser.add_argument('--img_size',           type=int,  default=64)
parser.add_argument('--self_condition',     type=bool, default=True)
parser.add_argument('--dim_mults',          type=int,  nargs='+', default=[1, 2, 4, 8])
# Diffusion
parser.add_argument('--timesteps',          type=int,  default=1000)
parser.add_argument('--sampling_timesteps', type=int,  default=150)
parser.add_argument('--beta_schedule',      type=str,  default='linear')
# Testing
parser.add_argument('--checkpoint',         type=str,  default='./results/model-8.pt')
parser.add_argument('--save_name',          type=str,  default='test_image')
parser.add_argument('--data_folder',        type=str,  default='/cluster/project7/backup_masramon/IQT/PICAI/')
parser.add_argument('--batch_size',         type=int,  default=5)
parser.add_argument('--is_pretrain',        action='store_true')
parser.add_argument('--finetune',           dest='is_pretrain', action='store_false')
parser.set_defaults(is_pretrain=True)
args, unparsed = parser.parse_known_args()
 
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
