import torch
import os
import math
import argparse
from tqdm import tqdm
from glob import glob
from utils import load_yaml_config, K15_dataset, StereoWaterdrop_dataset,StereoSR_dataset, chw_to_hwc, write_img,  evaluate_psnr_ssim,\
                    SOPMonitor, Conv1x1, Conv3x3, Linear, SpikingMatmul, count_convNd, count_linear, count_matmul,get_logger,evaluate_psnr_ssim_ms_ssim_lpips
from torch.utils.data import DataLoader
from natsort import natsorted
from models import create_model
from spikingjelly.activation_based import functional,base
import numpy as np
import pandas as pd
from thop import profile
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./configs/unet5_wwLifMul_1CAM_snn_sr_32to160_vth02_SRBB_SSCM_SSCA_SSRB_1e3_4_c32.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = load_yaml_config(args.opt)

    logger = get_logger(opt['model_name'], natsorted(glob(os.path.join(opt['path']['saved_path'], '*%s' % '.log')))[-1])

    ### gpu num
    device = str(opt['datasets']['test']['gpu_id'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    ### create model
    model = create_model(opt).cuda()
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids)>1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    path_chk_rest = natsorted(glob(os.path.join(opt['path']['saved_path'], '*%s' % 'model_best.pth')))[-1]
    checkpoint = torch.load(path_chk_rest)
    model.load_state_dict(checkpoint)

    ### SOPs and param
    model.eval()
    # mon = SOPMonitor(model)
    # mon.enable()

    tmp_path = opt['datasets']['test']['dataroot']
    for dataset_class in ['Flickr1024','KITTI2012', 'KITTI2015','Middlebury']:
        # opt['datasets']['test']['dataroot'] = './data/NAFSSR/test/{}/lr_x2'.format(dataset_class)
        opt['datasets']['test']['dataroot'] = tmp_path.format(dataset_class)

        results_dir = os.path.join(opt['path']['saved_path'], 'results_{}'.format(dataset_class))
        os.makedirs(results_dir, exist_ok=True)


        ### dataloader
        test_dataset_opt = opt['datasets']['test']
        if opt['datasets']['kind'] == 'k15':
            test_dataset = K15_dataset(test_dataset_opt['dataroot'], mode='test')
            test_loader = DataLoader(test_dataset, batch_size=test_dataset_opt['batch_size'],
                                     num_workers=0, pin_memory=True)
        elif opt['datasets']['kind'] == 'StereoWaterdrop':
            test_dataset = StereoWaterdrop_dataset(test_dataset_opt['dataroot'], mode='test')
            test_loader = DataLoader(test_dataset, batch_size=test_dataset_opt['batch_size'],
                                     num_workers=0, pin_memory=True)
        elif opt['datasets']['kind'] == 'StereoSR':
            test_dataset = StereoSR_dataset(test_dataset_opt['dataroot'], scale=opt['network']['sr_scale'], mode='test')
            test_loader = DataLoader(test_dataset, batch_size=test_dataset_opt['batch_size'],
                                     num_workers=0, pin_memory=True)


        img_multiple_of = 16
        for i, data in enumerate(tqdm(test_loader, unit='img'), 0):
            lq_l = data['lq_l'].cuda()
            B, C, H, W = lq_l.shape
            gt_l = data['gt_l'].cuda()
            lq_r = data['lq_r'].cuda()
            gt_r = data['gt_r'].cuda()
            names_l = data['name_l']
            names_r = data['name_r']

            with torch.no_grad():
                restored_l, restored_r = model(lq_l, lq_r)

                ######
                functional.reset_net(model)
                ######
                # [-1, 1] to [0, 1]
                restored_l = restored_l.clamp_(0, 1)
                restored_r = restored_r.clamp_(0, 1)


            # save
            os.makedirs(os.path.join(results_dir, 'restored_l'), exist_ok=True)
            os.makedirs(os.path.join(results_dir, 'restored_r'), exist_ok=True)
            for res_l, res_r, name_l, name_r in zip(restored_l, restored_r, names_l, names_r):

                rst_l = chw_to_hwc(res_l.detach().cpu().numpy())
                write_img(os.path.join(results_dir, 'restored_l', name_l), rst_l)
                res_r = chw_to_hwc(res_r.detach().cpu().numpy())
                write_img(os.path.join(results_dir, 'restored_r', name_r), res_r)

        # calculate psnr and ssim, save restored images
        if opt['datasets']['kind'] == 'k15':
            PSNRs_l, SSIMs_l, gt_names_l, _ = evaluate_psnr_ssim(
                os.path.join(test_dataset_opt['dataroot'], 'image_2_3_norain'), os.path.join(results_dir, 'restored_l'))
            PSNRs_r, SSIMs_r, gt_names_r, _ = evaluate_psnr_ssim(
                os.path.join(test_dataset_opt['dataroot'], 'image_3_2_norain'), os.path.join(results_dir, 'restored_r'))
        elif opt['datasets']['kind'] == 'StereoWaterdrop' or  opt['datasets']['kind'] == 'StereoSR':
            PSNRs_l, SSIMs_l, gt_names_l, _ = evaluate_psnr_ssim(
                os.path.join(test_dataset_opt['dataroot'], 'gt', 'left'), os.path.join(results_dir, 'restored_l'))
            PSNRs_r, SSIMs_r, gt_names_r, _ = evaluate_psnr_ssim(
                os.path.join(test_dataset_opt['dataroot'], 'gt', 'right'), os.path.join(results_dir, 'restored_r'))
        PSNRs = (np.array(PSNRs_l) + np.array(PSNRs_r)) / 2
        SSIMs = (np.array(SSIMs_l) + np.array(SSIMs_r)) / 2
        PSNR_mean, PSNR_std = np.mean(PSNRs), np.std(PSNRs, ddof=1)
        SSIM_mean, SSIM_std = np.mean(SSIMs), np.std(SSIMs, ddof=1)

        data = {
            'left_right': [l + r for l, r in zip(gt_names_l, gt_names_r)],
            'PSNR': PSNRs,
            'SSIM': SSIMs,
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_dir, '%.4f | %.4f.csv' % (PSNR_mean, SSIM_mean)), index=False)

        logger.info("[PSNR] mean: {:.4f} std: {:.4f}".format(PSNR_mean, PSNR_std))
        logger.info("[SSIM] mean: {:.4f} std: {:.4f}".format(SSIM_mean, SSIM_std))