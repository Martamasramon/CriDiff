from PIL import Image
from torchvision        import transforms as T, utils
from torch.utils.data   import Dataset
import pandas as pd
import numpy  as np
import torch

import sys
import os
sys.path.append(os.path.abspath('../../UNet/runet_t2w'))
from runetv2 import RUNet

class MyDataset(Dataset):
    def __init__(
        self,
        img_path,
        image_size      = 64,
        use_histo       = False, 
        use_T2W         = False, 
        is_finetune     = False, 
        surgical_only   = False,
        is_train        = True, 
        t2w_model_drop  = [0.1,0.5],
        t2w_model_path  = '/cluster/project7/ProsRegNet_CellCount/UNet/checkpoints/checkpoints_0306_1947_stage_1_best.pth'
    ):
        super().__init__() ## Necessary? Not in other...
        
        root   = 'finetune' if is_finetune else 'pretrain'
        suffix = 'train'    if is_train    else 'test'
        if surgical_only:
            root += '_surgical'
            
        self.img_path   = img_path
        self.img_dict   = pd.read_csv(f'../../Dataset_preparation/{root}_{suffix}.csv')
        self.image_size = image_size
        self.use_histo  = use_histo
        self.use_T2W    = use_T2W
        
        self.low_res_transform = T.Compose([
            T.CenterCrop(image_size),
            T.Resize(image_size//2, interpolation=T.InterpolationMode.NEAREST),
            T.Resize(image_size,    interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ])
        
        self.high_res_transform = T.Compose([
            T.CenterCrop(image_size),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ]) 
        
        if use_T2W:
            # Load pre-trained T2W embedding model
            self.t2w_model = RUNet(t2w_model_drop[0], t2w_model_drop[1])
            self.t2w_model.load_state_dict(torch.load(t2w_model_path))
            self.t2w_model.eval() 
            
            self.t2w_transform = T.Compose([
                T.CenterCrop(image_size),
                T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255) #T.ToTensor(), 255???
            ]) 
                                    
    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        item   = self.img_dict.iloc[idx]
        img    = Image.open(f'{self.img_path}/ADC/{item["SID"]}').convert('L')
        sample = {}

        if self.use_histo:
            embed_histo     = torch.load(item["Histo"],weights_only=True) 
            sample['Histo'] = embed_histo
        
        if self.use_T2W:
            t2w           = Image.open(f'{self.img_path}/T2W/{item["SID"]}').convert('L')
            t2w_input     = self.t2w_transform(t2w)
            sample['T2W'] = self.t2w_model.get_all_embeddings(t2w_input)
            
        sample['LowRes']  = self.low_res_transform(img)
        sample['HighRes'] = self.high_res_transform(img)

        return sample
    