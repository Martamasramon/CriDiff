import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from tqdm.auto import tqdm

import sys
sys.path.append('../module')
from network_utils   import *
from network_modules import *

from ema_pytorch import EMA


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataloader,
        accelerator,
        *,
        use_T2W                     = False,
        use_histo                   = False,
        batch_size                  = 16,
        gradient_accumulate_every   = 1,
        # augment_horizontal_flip     = True,
        lr                          = 1e-4,
        train_num_steps             = 100000,
        ema_update_every            = 10,
        ema_decay                   = 0.995,
        adam_betas                  = (0.9, 0.99),
        save_every                  = 5000,
        sample_every                = 2000,
        num_samples                 = 25,
        results_folder              = './results',
        amp                         = False,
        mixed_precision_type        = 'fp16',
        split_batches               = True,
        inception_block_idx         = 2048,
        max_grad_norm               = 1.,
        save_best_and_latest_only   = False
    ):
        super().__init__()

        self.accelerator    = accelerator
        self.model          = diffusion_model
        self.channels       = diffusion_model.input_img_channels
        is_ddim_sampling    = diffusion_model.is_ddim_sampling
        
        self.use_histo  = use_histo
        self.use_T2W    = use_T2W

        # # default convert_image_to depending on channels
        # if not exists(convert_image_to):
        #     convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples    = num_samples
        self.save_every     = save_every
        self.sample_every   = sample_every

        self.batch_size                = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps    = train_num_steps
        self.image_size         = diffusion_model.image_size
        self.max_grad_norm      = max_grad_norm

        dataloader      = self.accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        self.opt = Adam(diffusion_model.parameters(), lr = lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # from apex import amp
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if save_best_and_latest_only:
            self.best_mse = 1

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step':     self.step,
            'model':    self.accelerator.get_state_dict(self.model),
            'opt':      self.opt.state_dict(),
            'ema':      self.ema.state_dict(),
            'scaler':   self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def calc_loss(self, ):
        total_loss, total_mse, total_percpt, total_ssim = 0.0, 0.0, 0.0, 0.0

        for _ in range(self.gradient_accumulate_every):
            data = next(self.dataloader)
            
            for key, value in data.items():
                try:
                    data[key] = value.to(self.accelerator.device)
                except:
                    data[key] = [i.to(self.accelerator.device) for i in value]
                     
            with self.accelerator.autocast():
                if self.use_T2W and self.use_histo:
                    loss, mse, perct, ssim = self.model(data['HighRes'], data['LowRes'], t2w=data['T2W'], histo=data['Histo'])
                elif self.use_T2W:
                    data['T2W'] = [t.squeeze(1) for t in data['T2W']]
                    loss, mse, perct, ssim = self.model(data['HighRes'], data['LowRes'], t2w=data['T2W'])
                elif self.use_histo:
                    loss, mse, perct, ssim = self.model(data['HighRes'], data['LowRes'], histo=data['Histo'])
                else:
                    loss, mse, perct, ssim = self.model(data['HighRes'], data['LowRes'])
                
                total_loss   += loss.item()  / self.gradient_accumulate_every
                total_mse    += mse.item()   / self.gradient_accumulate_every
                total_percpt += perct.item() / self.gradient_accumulate_every
                total_ssim   += ssim.item()  / self.gradient_accumulate_every

            self.accelerator.backward(loss)
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(name)
                    
        return data, total_loss, total_mse, total_percpt, total_ssim
                    
    def sample_images(self, data):
        with torch.no_grad():
            milestone       = self.step // self.sample_every
            batches         = num_to_groups(self.num_samples, self.batch_size)
            sample_lowres   = data['LowRes'][:self.num_samples].to(self.accelerator.device)
            
            # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, cond=sample_lowres), batches))
            all_images_list = []
            start = 0
            total = sample_lowres.shape[0]

            for n in batches:
                end = min(start + n, total)
                cond_slice = sample_lowres[start:end]
                
                if cond_slice.shape[0] == 0:
                    break  # no more valid conditioning inputs

                images = self.ema.ema_model.sample(batch_size=cond_slice.shape[0], low_res=cond_slice)
                all_images_list.append(images)
                start = end
        
        all_images = torch.cat(all_images_list, dim = 0)
        
        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                
    def train(self):
        accelerator = self.accelerator

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                # Calculate loss
                data, total_loss, total_mse, total_percpt, total_ssim = self.calc_loss()
                
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # Sample images
                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()
                        self.sample_images(data)
                        
                    # Save model 
                    if self.step != 0 and divisible_by(self.step, self.save_every):
                        milestone = self.step // self.save_every
                        if self.save_best_and_latest_only:
                            if self.best_mse > total_mse:
                                self.best_mse = total_mse
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)
                
                # Update pbar
                if self.step % 100 == 0:
                    pbar.set_description(f"loss: {total_loss:.4f} (MSE: {total_mse:.4f},  perct: {total_percpt:.4f}, SSIM: {total_ssim:.4f},)")
                    pbar.update(100)

        accelerator.print('training complete')
