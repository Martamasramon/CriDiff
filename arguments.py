import argparse

parser = argparse.ArgumentParser("Diffusion")
# UNet
parser.add_argument('--img_size',           type=int,  default=64)
parser.add_argument('--down',               type=int,  default=2)
parser.add_argument('--self_condition',     type=bool, default=True)
parser.add_argument('--dim_mults',          type=int,  nargs='+', default=[1, 2, 4, 8])
# Diffusion
parser.add_argument('--timesteps',          type=int,  default=1000)
parser.add_argument('--sampling_timesteps', type=int,  default=150)
parser.add_argument('--beta_schedule',      type=str,  default='linear')
parser.add_argument('--perct_λ',            type=float,default=0.1)
# Training
parser.add_argument('--results_folder',     type=str,  default='./results')
parser.add_argument('--batch_size',         type=int,  default=16)
parser.add_argument('--lr',                 type=float,default=8e-5)
parser.add_argument('--n_epochs',           type=int,  default=40000)
parser.add_argument('--ema_decay',          type=float,default=0.995)
# Log process
parser.add_argument('--save_every',         type=int,  default=2000)
parser.add_argument('--sample_every',       type=int,  default=2000)
# Test params
parser.add_argument('--checkpoint',         type=str,  default=None) #'./results_perct/model-8.pt'
parser.add_argument('--save_name',          type=str,  default=None)
# Bools
parser.add_argument('--use_T2W',            action='store_true')
parser.add_argument('--use_histo',          action='store_true')
parser.add_argument('--use_mask',           action='store_true')
parser.add_argument('--surgical_only',      action='store_true')
parser.add_argument('--finetune',           action='store_true') # Use HistoMRI dataset
parser.add_argument('--controlnet',         action='store_true') 
parser.set_defaults(use_T2W       = False)
parser.set_defaults(use_histo     = False)
parser.set_defaults(use_mask      = False)
parser.set_defaults(surgical_only = False)
parser.set_defaults(finetune      = False)
parser.set_defaults(controlnet    = False)

args, unparsed = parser.parse_known_args()
