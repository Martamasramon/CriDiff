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
from Diffusion_Basic import Diffusion_Basic
from UNet_Basic      import UNet_Basic

import sys
sys.path.append('../')
from dataset         import MyDataset
from test_functions  import *
from arguments       import args

folder = '/cluster/project7/backup_masramon/IQT/'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
 
def main():
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
        beta_schedule       = args.beta_schedule,
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
        folder, 
        args.img_size, 
        data_type       = 'val', 
        is_finetune     = args.finetune,
        use_mask        = args.use_mask, 
        downsample      = args.down
    ) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print('Visualising...')
    save_name = args.save_name if args.save_name is not None else os.path.basename(os.path.dirname(args.checkpoint))
    test_data = 'HistoMRI' if args.finetune else 'PICAI'
    
    visualize_batch(diffusion, dataloader, args.batch_size, device, output_name=f'{save_name}_{test_data}')
    
    print('Evaluating...')
    evaluate_results(diffusion, dataloader, device, args.batch_size)
    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
