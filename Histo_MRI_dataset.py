import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageStat
from torchvision import transforms as T
from functools import partial
from pathlib import Path
from torch import nn
import albumentations as A
import cv2
import re

def sort_by_number(path):
    numbers = re.findall(r'\d+', str(path))
    return int(numbers[-1]) if numbers else 0

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def exists(x):
    return x is not None


class Dataset(data.Dataset):
    def __init__(
            self,
            folder,
            image_size,
            mode,
            convert_image_to,
            downsample_scale = 4,
            exts=['jpg', 'jpeg', 'png', 'tiff'],

    ):
        super().__init__()
        self.folder           = folder
        self.image_size       = image_size
        self.mode             = mode
        self.convert_image_to = convert_image_to
        self.downsample_scale = downsample_scale
        
        assert mode == 'train' or mode == 'test'
        if mode == 'train':
            self.transform = A.Compose([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5, rotate_limit=90, border_mode=0, value=0, p=0.5),
                A.Resize(height=320, width=320, interpolation=cv2.INTER_NEAREST),
                A.RandomCrop(height=image_size, width=image_size),
            ], additional_targets={'body': 'mask', 'detail': 'mask'})
            
        else:
            self.transform = A.Compose([
                A.Resize(height=256, width=256),
            ])
        
        self.img_paths  = sorted([p for ext in exts for p in Path(f'{img_paths}').glob(f'**/*.{ext}')], key=sort_by_number)
        self.mask_paths = sorted([p for ext in exts for p in Path(f'{mask_paths}').glob(f'**/*.{ext}')], key=sort_by_number)
    


    def downsample(image, scale, mantain_size=True):
        # FOR SHAPE (B, H, W, C)
        h, w         = image.shape[1:3]
        new_h, new_w = h // scale, w // scale
        
        if mantain_size==True:
            result = image.copy()    
            for i in range(image.shape[0]):
                down_lowres     = cv2.resize(image[i,:,:,:], (new_w, new_h), interpolation=cv2.INTER_AREA)
                result[i,:,:,:] = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            result = np.zeros((image.shape[0], new_h, new_w, image.shape[3]))
            for i in range(image.shape[0]):
                result[i,:,:,:] = cv2.resize(image[i,:,:,:], (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return result

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_paths = self.img_paths[index]
            img = Image.open(img_paths)
            img = convert_image_to_fn(self.convert_image_to, img)
            
            # t2w_embedding  = 
            # histo_embedding =

            transform  = self.transform(image=np.array(img))
            image_data = torch.from_numpy(transform['image']).float().unsqueeze(dim=0)

            return image_data

        elif self.mode == 'test':
            img_paths = self.img_paths[index]
            img = Image.open(img_paths)
            img = convert_image_to_fn(self.convert_image_to, img)

            transform = self.transform(image=np.array(img), mask=np.array(mask))
            image_data = transform['image']
            
            #image_name = str(self.img_paths[index]).split('/')[-1].split('.')[0]
            image_data = torch.from_numpy(image_data).float().unsqueeze(dim=0)
            
            return image_data #, image_name

        else:
            raise ValueError


if __name__ == '__main__':

    train_dataset = Dataset("/home/ubuntu/data/ProstateV2", 256, 'train', convert_image_to='L')
    test_dataset = Dataset("/home/ubuntu/data/ProstateV2", 256, 'test', convert_image_to='L')

    test_que = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, drop_last=False,
        pin_memory=True, shuffle=True)

    train_qus = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, drop_last=False,
        pin_memory=True, shuffle=True)

    for i, (data) in enumerate(train_qus):
        # image_data, mask_data, name = data
        image_data, mask_data, body_data, detail_data = data
        print(i)
