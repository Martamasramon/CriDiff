from PIL import Image
from transforms         import get_transforms
from torch.utils.data   import Dataset
import pandas as pd
import numpy  as np
import torch

import sys
import os
sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet

class MyDataset(Dataset):
    def __init__(
        self,
        img_path,
        data_type       = None, 
        blank_prob      = 0,
        image_size      = 64,
        t2w             = False, 
        t2w_embed       = False, 
        is_finetune     = False, 
        surgical_only   = False,
        t2w_model_drop  = [0.1,0.5],
        t2w_model_path  = '/cluster/project7/ProsRegNet_CellCount/UNet/checkpoints/default_64.pth',
        use_mask        = False, 
        downsample      = 2,
    ):
        super().__init__() ## Necessary? Not in other...
        
        root   = 'finetune' if is_finetune else 'pretrain'
        if surgical_only:
            root += '_surgical'
        
        self.masked     = '_mask' if use_mask else ''
        self.img_path   = img_path + 'HistoMRI/target_adc' if is_finetune else img_path + 'PICAI' 
        self.img_dict   = pd.read_csv(f'/cluster/project7/ProsRegNet_CellCount/Dataset_preparation/CSV/{root}{self.masked}_{data_type}.csv')
        self.t2w_embed  = t2w_embed
        self.use_T2W    = t2w or t2w_embed
        self.data_type  = data_type
        self.blank_prob = blank_prob
        self.low_res_transform, self.high_res_transform, self.t2w_transform = get_transforms(2, image_size, downsample)
        
        if self.use_T2W:
            # Load pre-trained T2W embedding model
            self.t2w_model = RUNet(t2w_model_drop[0], t2w_model_drop[1], img_size=image_size)
            self.t2w_model.load_state_dict(torch.load(t2w_model_path))
            self.t2w_model.eval() 
                                            
    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        item   = self.img_dict.iloc[idx]
        img    = Image.open(f'{self.img_path}/ADC{self.masked}/{item["SID"]}').convert('L')
        sample = {}
        
        if self.use_T2W:
            t2w           = Image.open(f'{self.img_path}/T2W{self.masked}/{item["SID"]}').convert('L')
            # t2w          = Image.new('L', t2w.size, 0) # TEst performance with blank image
            sample['T2W'] = self.t2w_transform(t2w)
            
        if self.t2w_embed:
            sample['T2W_embed'] = self.t2w_model.get_all_embeddings(sample['T2W'].unsqueeze(0))
        
        sample['LowRes']  = self.low_res_transform(img)
        sample['HighRes'] = self.high_res_transform(img)
        
        #### Force T2W usage by sometimes deleting DWI input
        if self.data_type == 'train' and torch.rand(()) < self.blank_prob:
            sample['LowRes'] = torch.zeros_like(sample['LowRes'])

        return sample
    

class MyDataset3D(Dataset):
    def __init__(
        self,
        img_path,
        data_type       = None, 
        blank_prob      = 0,
        image_size      = 64,
        t2w             = False, 
        is_finetune     = False, 
        use_mask        = False, 
        downsample      = 2,
    ):
        super().__init__() 
        
        root   = 'finetune' if is_finetune else 'pretrain'
        
        # self.masked     = '_mask' if use_mask else ''
        self.img_path   = img_path + 'HistoMRI' if is_finetune else img_path + 'PICAI' 
        self.img_dict   = pd.read_csv(f'/cluster/project7/ProsRegNet_CellCount/Dataset_preparation/3d_PI-CAI_{data_type}.csv')
        self.t2w_embed  = t2w_embed
        self.use_T2W    = t2w 
        self.data_type  = data_type
        self.blank_prob = blank_prob
        self.low_res_transform, self.high_res_transform, self.t2w_transform = get_transforms(3, image_size, downsample)
                                            
    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        item   = self.img_dict.iloc[idx]
        img    = sitk.GetArrayFromImage(sitk.ReadImage(f'{self.img_path}/{item["ADC"]}'))
        sample = {}
        
        if self.use_T2W:
            t2w           = sitk.GetArrayFromImage(sitk.ReadImage(f'{self.img_path}/{item["T2W"]}'))
            sample['T2W'] = self.t2w_transform(t2w)
        
        sample['LowRes']  = self.low_res_transform(img)
        sample['HighRes'] = self.high_res_transform(img)
        
        #### Force T2W usage by sometimes deleting DWI input
        if self.data_type == 'train' and torch.rand(()) < self.blank_prob:
            sample['LowRes'] = torch.zeros_like(sample['LowRes'])

        return sample
    