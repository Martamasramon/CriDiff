import torch
import argparse
import os
import logging
import sys
import cv2
import time
import utils
import re

from skimage                        import img_as_ubyte
from torch.utils.data.distributed   import DistributedSampler
from metrics                        import evaluate_single
from module.DiffusionModel          import DiffSOD
from Prostate_dataset               import Dataset
from collections                    import OrderedDict

parser = argparse.ArgumentParser("Diffusion infer")
# Required
# Path save Eval log.
parser.add_argument("--loadDir", type=str, default='./exp/20240221-133240_5e-05_linear_PVT_GLenhanceV21_Diff_dim64_Prostate_init')
# Path load ck files.
parser.add_argument("--checkpoint", type=str, default='./checkpoints/pvt_v2_b2.pth')
parser.add_argument("--beta_sched", type=str, default='linear', help='cosine or linear')
parser.add_argument("--num_timesteps", type=float, default=500, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl")
parser.add_argument('--size', type=int, default=256, help='test_size')
parser.add_argument('--dataset_name', type=str, default='PROMISE12', help='test_size')
parser.add_argument('--dataset_root', type=str, default='/home/oip/data/ProstateSeg/PROMISE12', help='note for this run')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--num_ens', type=int, default=25,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
parser.add_argument('--sampling_timesteps', type=int, default=30,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')

# Generation parametersa
parser.add_argument("--device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
parser.add_argument("--self_condition", type=bool, default=True, help="self_condition", required=False)

# Output parameters
parser.add_argument("--print_freq", type=int, default=50, required=False)
parser.add_argument('--job_name', type=str, default='IQT', help='job_name')

args, unparsed = parser.parse_known_args()

if not os.path.exists(args.job_name):
    os.makedirs(args.job_name)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1 

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract_number(f):
    s = re.findall(r'\d+', f)
    return (int(s[0]) if s else -1, f)
    
def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0) 

    # train_dataset = Dataset(args.dataset_root, args.size, 'train', convert_image_to='L')
    test_dataset  = Dataset(args.dataset_root, args.size, 'test', convert_image_to='L')

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0,
        pin_memory=True, shuffle=False)

    # Create model
    diffusion = DiffSOD(args, sampling_timesteps=args.sampling_timesteps if args.sampling_timesteps > 0 else None)
    diffusion = diffusion.to(device)

    # Load checkpoint to model
    save_dict  = torch.load(args.checkpoint)
    new_state_dict = OrderedDict()
    for k, v in save_dict['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k  
        new_state_dict[name] = v
    diffusion.load_state_dict(new_state_dict, strict=True)


    score_metricsC = epoch_evaluating(diffusion, args.checkpoint, test_dataloader, device, evaluate_single)
    fg_dice         = score_metricsC["f1"].item()       # baseline 80 sota 86
    score_iou_poly  = score_metricsC["iou_poly"].item()
    score_avg_msd   = score_metricsC["avg_msd"].item()  # baseline 2.16 sota 1.96
    score_avg_asd   = score_metricsC["avg_asd"].item()  # baseline 2.16 sota 1.96

    print('dataset_name {}'.format(args.dataset_name))
    print('FG_Dice %f',     fg_dice)
    print('Iou_poly %f',    score_iou_poly)
    print('MSD %f',         score_avg_msd)
    print('ASD %f',         score_avg_asd)


def epoch_evaluating(model, checkpoint_file, test_dataloader, device, criteria_metrics):
    # Switch model to evaluation mode
    model.eval()
    out_pred_final = torch.FloatTensor().cuda(device)
    out_gt         = torch.FloatTensor().cuda(device)  # Tensor stores groundtruth values
    
    with torch.no_grad():  # Turn off gradient
        
        # For each batch
        for step, (images, masks, index) in enumerate(test_dataloader):
            # Move images, labels to device (GPU)
            input_img = images.cuda(device)
            masks     = masks.cuda(device)
            # input = images * 2 - 1
            preds = torch.zeros((input_img.shape[0], args.num_ens, input_img.shape[2], input_img.shape[3])).cuda(device)
            for i in range(args.num_ens):
                preds[:, i:i + 1, :, :] = model.sample(input_img)
            preds_mean = preds.mean(dim=1)
            # out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            # preds_mean = preds_mean
            preds_mean[preds_mean < 0.3] = 0
            # preds_mean[preds_mean > 1] = 1

            # preds_mean1 = preds_mean.data.cpu().numpy()
            out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            out_gt = torch.cat((out_gt, masks), 0)
            # masks = torch.squeeze(masks)
            # preds_std = preds.std(dim=1)
            for idx in range(preds.shape[0]):
                predict_rgb = preds_mean[idx, :, :].cpu().detach()
                # # predict_rgb = torch.squeeze(predict_rgb)
                # predict_rgb = predict_rgb.sigmoid().numpy()
                # predict_rgb = (predict_rgb / predict_rgb.max()).cpu().detach().numpy()
                predict_rgb = img_as_ubyte(predict_rgb)
                
                # for i in range(input_img.shape[0]):
                cv2.imwrite(args.job_name + '/' + str(index[idx]) + '.png', predict_rgb)

                # preds_single_eval = preds_mean[idx, :, :].unsqueeze(dim=0)
                # _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
                #     preds_single_eval, masks[idx])
                # logging.info()
                # preds_single_eval_1 = preds_single_eval.cpu().numpy()
                # mask_1=masks[idx].cpu().numpy()
                # if preds_single_eval_1.sum() != 0 and mask_1.sum() != 0:
                #     dice = metric.binary.dc(preds_single_eval_1, mask_1)
                #     iou = metric.binary.jc(preds_single_eval_1, mask_1)
                #     msd = metric.binary.hd95(preds_single_eval_1, mask_1)
                #     logging.info('ID {}'.format(str(index[idx])))
                #     logging.info('Dice %f', dice)
                #     logging.info('Iou %f', iou)
                #     logging.info('Msd %f', msd)
                # iou = metric.binary.i
                # logging.info('ID {}'.format(str(index[idx])))
                # logging.info('FG_Dice %f', _F1C)
                # logging.info('Iou_poly %f', _IoU_polyC)
                # logging.info('Iou_mean %f', _IoU_meanC)
                # logging.info('MSD %f', _MSD)
                # logging.info('ASD %f', _ASD)
            if step % args.print_freq == 0 or step == len(test_dataloader) - 1:
                logging.info(
                    "val: Step {:03d}/{:03d}".format(step, len(test_dataloader) - 1))

    _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
        out_pred_final, out_gt)

    score_metricsC = {
        "recall": _recallC,
        "specificity": _specificityC,
        "precision": _precisionC,
        "f1": _F1C,
        "f2": _F2C,
        "accuracy": _ACC_overallC,
        "iou_poly": _IoU_polyC,
        "iou_bg": _IoU_bgC,
        "iou_mean": _IoU_meanC,
        "avg_msd":_MSD,
        "avg_asd": _ASD,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return validation loss, and metric score
    return score_metricsC
if __name__ == '__main__':
    main()
