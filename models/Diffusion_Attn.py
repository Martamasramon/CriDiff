import torch
import torch.nn.functional as F
from torch          import nn
from random         import random
from functools      import partial
from collections    import namedtuple
from einops         import rearrange
from tqdm.auto      import tqdm
import numpy        as np

from network_utils   import *
from Diffusion_Basic import Diffusion_Basic

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class Diffusion_Attn(Diffusion_Basic):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def model_predictions(self, c, x, t, x_self_cond=None, clip_x_start=False):
        model_output, _, _, _ = self.model(c, x, t,  x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

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


    def p_mean_variance(self, c, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(c, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start


    @torch.no_grad()
    def p_sample(self, c, x, t,x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(c=c, x=x, t=batched_times,
                                                                          x_self_cond=x_self_cond,
                                                                          clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device   = shape[0], self.betas.device
        img             = torch.randn(shape, device=device)
        x_start         = None

        # batched_times = torch.full((img.shape[0],), 10, device=img.device, dtype=torch.long)
        # model_out, input_side_out, _, _ = self.model(cond, img, batched_times, None)

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(cond, img, t, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img, input_side_out[0]


    @torch.no_grad()
    def ddim_sample(self, shape, cond_img, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None
        batched_times = torch.full((img.shape[0],), 10, device=img.device, dtype=torch.long)
        model_out, input_side_out, body_pre, detail_pre = self.model(cond_img, img, batched_times, None)
        gif = []
        gif_t = [0, 6, 12, 15, 18, 21, 24, 26,30]
        gif_time = list(times[i] for _,i in enumerate(gif_t))
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(cond_img, img, time_cond, self_cond,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            if time in gif_time and time>0:
                gif.append(img)

        img = unnormalize_to_zero_to_one(img)
        gif.append(img)
        return img, gif


    @torch.no_grad()
    def sample(self, cond_img):
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)

        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        b1 = noise.data.cpu().detach().numpy()

        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def p_losses(self, x_start, t, cond, noise=None):
        b, c, h, w  = x_start.shape
        noise       = default(noise, lambda: torch.randn_like(x_start))
        noise       = noise.cuda()
        x           = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times and condition 
        # with unet with that this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # predicting x_0
                x_self_cond = self.model_predictions(cond, x, t).pred_x_start
                x_self_cond.detach_()

        model_out, input_side_out, body_pre, detail_pre = self.model(cond, x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target)
        from loss import lossFunct, structure_loss, comput_loss
        x_start = unnormalize_to_zero_to_one(x_start)
       
        return loss.mean(), input_side_out, body_pre, detail_pre


    def forward(self, cond_img, mask, *args, **kwargs):
        if mask.ndim == 3:
            mask = rearrange(mask, 'b h w -> b 1 h w')

        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        device = self.device
        mask, cond_img = mask.to(device), cond_img.to(device)

        b, c, h, w, device, img_size, img_channels, mask_channels = *mask.shape, mask.device, self.image_size, self.input_img_channels, self.mask_channels

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert cond_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        assert mask.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        img = normalize_to_neg_one_to_one(mask)
        return self.p_losses(img, times, cond_img, *args, **kwargs)
