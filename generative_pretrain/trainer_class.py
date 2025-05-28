import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset

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
        input_folder,
        accelerator,
        *,
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
        calculate_fid               = True,
        inception_block_idx         = 2048,
        max_grad_norm               = 1.,
        num_fid_samples             = 50000,
        save_best_and_latest_only   = False
    ):
        super().__init__()

        self.accelerator    = accelerator
        self.model          = diffusion_model
        self.channels       = diffusion_model.input_img_channels
        is_ddim_sampling    = diffusion_model.is_ddim_sampling

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

        # dataset and dataloader
        self.ds     = MyDataset(input_folder, self.image_size) 
        dl          = DataLoader(self.ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)
        dl          = self.accelerator.prepare(dl)
        self.dl     = cycle(dl)

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

        # FID-score computation
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

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

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = {k: v.to(device) for k, v in data.items()}

                    with self.accelerator.autocast():
                        loss = self.model(data['HighRes'], cond=data['LowRes'])
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(name)

                # pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            sample_lowres = data['LowRes'][:self.num_samples].to(device)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, cond=sample_lowres), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                    if self.step != 0 and divisible_by(self.step, self.save_every):
                        milestone = self.step // self.save_every
                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                if self.step % 100 == 0:
                    pbar.update(100)
                    pbar.set_description(f"loss: {total_loss:.4f}")

        accelerator.print('training complete')
