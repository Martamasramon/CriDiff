import time
import torch
torch.cuda.set_device(0)
import argparse
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('../models')
from Diffusion_Attn import Diffusion_Attn
from UNet_Attn      import UNet_Attn
from network_utils  import *

import sys
sys.path.append('../')
from dataset         import MyDataset
from test_functions  import *

folder = '/cluster/project7/backup_masramon/IQT/'

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
parser.add_argument('--perct_λ',            type=float,default=0.1)
# Testing
parser.add_argument('--checkpoint',         type=str,  default='./results/model-8.pt')
parser.add_argument('--save_name',          type=str,  default='test_image')
parser.add_argument('--batch_size',         type=int,  default=15)
parser.add_argument('--finetune',           action='store_true')
parser.set_defaults(finetune = False)
# IQT
parser.add_argument('--use_T2W',            action='store_true')
parser.add_argument('--use_histo',          action='store_true')
parser.add_argument('--use_mask',           action='store_true')
parser.set_defaults(use_T2W   = False)
parser.set_defaults(use_histo = False)
parser.set_defaults(use_mask  = False)
args, unparsed = parser.parse_known_args()
 
def main():
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_folder = 'HistoMRI' if args.finetune else 'PICAI'
    
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

    print('Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])
    
    # Move model to device
    model.eval()
    model.to(device)
    diffusion.model = model
    diffusion.to(device)
    
    print('Loading data...')
    dataset     = MyDataset(
        folder + data_folder, 
        args.img_size, 
        data_type       = 'val', 
        is_finetune     = args.finetune,
        use_T2W         = args.use_T2W, 
        use_mask        = args.use_mask,
    ) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
 
    print('Visualising...')
    visualize_batch(diffusion, dataloader, args.batch_size, device, output_name=f'{args.save_name}_{data_folder}', use_T2W=args.use_T2W)
    
    print('Evaluating...')
    evaluate_results(diffusion, dataloader, device, args.batch_size, t2w=args.use_T2W)
   
    
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
