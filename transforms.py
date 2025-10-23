from torchvision        import transforms as T
import torch

import numpy as np
import torch
import torch.nn.functional as F

def _ensure_tuple3(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)

def to_tensor_3d_from_sitk(arr_3d):
    """
    SITK GetArrayFromImage -> numpy with shape (D,H,W). Convert to torch [1,H,W,D].
    """
    assert arr_3d.ndim == 3, f"Expected 3D array, got {arr_3d.shape}"
    D, H, W = arr_3d.shape
    x = np.nan_to_num(arr_3d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(x)                # [D,H,W]
    x = x.permute(1, 2, 0).contiguous()    # -> [H,W,D]
    x = x.unsqueeze(0)                     # -> [1,H,W,D]
    return x

def normalize_volume01(v: torch.Tensor, p_lo=1.0, p_hi=99.0):
    """
    Robust per-volume normalization to [0,1]. v: [1,H,W,D].
    """
    x = v.squeeze(0)  # [H,W,D]
    vals = x.flatten()
    lo = torch.quantile(vals, p_lo/100.0)
    hi = torch.quantile(vals, p_hi/100.0)
    if (hi - lo) > 0:
        x = (x - lo) / (hi - lo)
        x = x.clamp_(0, 1)
    else:
        # fallback: min-max
        mn = vals.min()
        mx = vals.max()
        x = (x - mn) / (mx - mn + 1e-6) if (mx > mn) else (x*0.0)
    return x.unsqueeze(0)  # [1,H,W,D]

def center_crop_3d(v: torch.Tensor, size):
    """
    Center-crop (with pad if needed). v: [1,H,W,D], size: int or (h,w,d).
    """
    h, w, d = _ensure_tuple3(size)
    _, H, W, D = v.shape

    sh = max(0, (H - h)//2); eh = sh + min(h, H)
    sw = max(0, (W - w)//2); ew = sw + min(w, W)
    sd = max(0, (D - d)//2); ed = sd + min(d, D)

    out = v[:, sh:eh, sw:ew, sd:ed]

    ph = max(0, h - out.shape[1])
    pw = max(0, w - out.shape[2])
    pd = max(0, d - out.shape[3])
    if ph or pw or pd:
        # pad: (D_right, D_left, W_right, W_left, H_right, H_left)
        pad = (0, pd, 0, pw, 0, ph)
        out = F.pad(out, pad, mode="constant", value=0.0)
    return out

def resize_3d(v: torch.Tensor, size):
    """
    v: [1,H,W,D] -> [1,h,w,d] with trilinear.
    """
    h, w, d = _ensure_tuple3(size)
    return F.interpolate(v.unsqueeze(0), size=(h, w, d), mode="trilinear", align_corners=False).squeeze(0)

def make_lowres_from_hr(hr: torch.Tensor, downsample: int):
    """
    hr: [1,h,w,d] -> downsample then upsample back to [1,h,w,d].
    """
    if downsample <= 1:
        return hr
    h, w, d = hr.shape[1:]
    ds = (max(1, h//downsample), max(1, w//downsample), max(1, d//downsample))
    lr = F.interpolate(hr.unsqueeze(0), size=ds, mode="trilinear", align_corners=False).squeeze(0)
    lr_up = F.interpolate(lr.unsqueeze(0), size=(h, w, d), mode="trilinear", align_corners=False).squeeze(0)
    return lr_up

def _pre(arr):
    # SITK (D,H,W) -> torch [1,H,W,D] normalized
    t = to_tensor_3d_from_sitk(arr)
    t = normalize_volume01(t)
    return t


class Transforms():
    def __init__(
        self,
        adc_size,
        downsample
    ):
        self.adc_size   = adc_size
        self.downsample = downsample
    
    def get_lowres(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Resize(self.adc_size//self.downsample, interpolation=T.InterpolationMode.NEAREST),
            T.Resize(self.adc_size,             interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ])
    def get_highres(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ]) 
    def get_t2w(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size*2),
            T.Resize(self.adc_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ]) 
    def get_transforms(self):
        return {
            'ADC_input'    : self.get_lowres(),
            'ADC_condition': self.get_highres(),
            'T2W_condition': self.get_t2w()
        }

class TransformsUpsample(Transforms):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_upsampled_nearest(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Resize(self.adc_size * self.downsample, interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ])
    def get_upsampled_bicubic(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Resize(self.adc_size * self.downsample, interpolation=T.InterpolationMode.BICUBIC),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ]) 
    def get_t2w(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size * self.downsample),
            T.ToTensor()
        ]) 
    def get_adc(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.ToTensor()
        ]) 
    def downsample_output(self):  
        return T.Compose([
            T.Resize(self.adc_size, interpolation=T.InterpolationMode.BICUBIC),
        ]) 
    def get_transforms(self):
        return {
            'ADC_input'    : self.get_upsampled_cubic(),
            'ADC_condition': self.get_upsampled_nearest(),
            'T2W_condition': self.get_t2w(),
            'ADC_target'   : self.get_adc(),
            'ADC_prediction':self.downsample_output()
        }
        
class Transforms3D(Transforms):
    def __init__(self, *args):
        super().__init__(*args)

    def get_lowres(self):
        return lambda arr: center_crop_3d(_pre(arr), self.image_size)
    def get_highres(self):
        return lambda arr: make_lowres_from_hr(center_crop_3d(_pre(arr), self.image_size), self.downsample)
    def get_t2w(self):  
        return lambda arr: center_crop_3d(_pre(arr), self.image_size)

def get_transforms(dims, image_size, downsample, upsample=False):
    if dims == 2:
        transform = TransformsUpsample(image_size, downsample) if upsample else Transforms(image_size, downsample)
    else:
        transform = Transforms3D(image_size, downsample)
        
    return transform.get_transforms()