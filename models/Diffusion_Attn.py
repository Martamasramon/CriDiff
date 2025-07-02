import torch
import torch.nn.functional as F
from torch          import nn
from random         import random
from functools      import partial
from collections    import namedtuple
from einops         import rearrange, reduce
from tqdm.auto      import tqdm
from piq            import ssim
import numpy        as np

from network_utils   import *
from Diffusion_Basic import Diffusion_Basic

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'x_start'])

class Diffusion_Attn(Diffusion_Basic):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def model_predictions(self, x, low_res, t, t2w=None, histo=None, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model.forward(x,low_res, t, t2w, histo, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)


    def p_mean_variance(self, x, low_res, t, t2w=None, histo=None, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, low_res, t, t2w, histo, x_self_cond)
        x_start = preds.x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start


    @torch.no_grad()
    def p_sample(self, x, low_res, t, t2w=None, histo=None, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, low_res, batched_times, t2w, histo, x_self_cond, clip_denoised=True)
        noise       = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img    = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    @torch.no_grad()
    def p_sample_loop(self, shape, low_res, t2w=None, histo=None, return_all_timesteps = False):
        batch, device = shape[0], self.device
        img           = torch.randn(shape, device = device)
        imgs          = [img]
        x_start       = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, low_res, t, t2w, histo, self_cond)
            imgs.append(img)

        img = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        img = self.unnormalize(img)
        return img


    @torch.no_grad()
    def ddim_sample(self, shape, low_res, t2w=None, histo=None, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img     = torch.randn(shape, device = device)
        imgs    = [img]
        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(img, low_res, time_cond, t2w, histo, self_cond, clip_x_start = True, rederive_pred_noise = True)
            x_start = preds.x_start
            
            if time_next < 0:
                img = preds.x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = preds.x_start * alpha_next.sqrt() + \
                  c * preds.pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret


    @torch.no_grad()
    def sample(self, low_res, t2w=None, histo=None, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn            = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), low_res, t2w, histo, return_all_timesteps = return_all_timesteps)


    
    def p_losses(self, x_start, low_res, t, t2w=None, histo=None, noise=None, offset_noise_strength = None):
        b, c, h, w              = x_start.shape
        noise                   = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength   = default(offset_noise_strength, self.offset_noise_strength)         # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
           
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # self-conditioning. 50% of the time, predict x_start from current set of times and condition with unet
        #                   slows down training by 25%, but seems to lower FID significantly
        
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x,low_res, t, t2w, histo).x_start  
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, low_res, t, t2w, histo, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise  
        elif self.objective == 'pred_x0':
            target = x_start  # Predict high-res directly
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
        mse_loss = F.mse_loss(model_out, target, reduction = 'none')
        mse_loss = reduce(mse_loss, 'b ... -> b', 'mean')
        
        perct_loss = self.perct_loss(model_out.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
        # print("VGG perceptual loss (batch):", perct_loss)
        # perct_loss = perct_loss.view(perct_loss.shape[0])
        
        with torch.no_grad():
            ssim_val   = ssim(model_out.clamp(0.0, 1.0), target.clamp(0.0, 1.0), data_range=1.0)
        
        loss = mse_loss + self.perct_λ * perct_loss
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean(), mse_loss.mean(), perct_loss.mean(), ssim_val.mean()


    def forward(self, img, low_res, t2w=None, histo=None, noise=None, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(self.normalize(img), low_res, t, t2w, histo, noise, **kwargs)
