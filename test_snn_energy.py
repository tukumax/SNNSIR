import torch
import os
import math
import argparse
from tqdm import tqdm
from glob import glob
from utils import load_yaml_config, K15_dataset, StereoWaterdrop_dataset, chw_to_hwc, write_img,  evaluate_psnr_ssim,\
                    SOPMonitor, Conv1x1, Conv3x3, Linear, SpikingMatmul, count_convNd, count_linear, count_matmul,get_logger
from torch.utils.data import DataLoader
from natsort import natsorted
from models import create_model
from spikingjelly.activation_based import functional,base
import numpy as np
import pandas as pd
from thop import profile

def splitimage(imgtensor, crop_size=80, overlap_size=8):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W)).cuda()
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=80, resolution=(1, 3, 80, 80)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W)).cuda()
    merge_img = torch.zeros((B, C, H, W)).cuda()
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./configs/unet5_wwLifMul_1CAM_snn_waterdrop_32to160_vth02_SRBB_SSCM_SSCA_SSRB_1e3_4_c32.yml',
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
    mon = SOPMonitor(model)
    mon.enable()

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

    results_dir = os.path.join(opt['path']['saved_path'], 'results')
    os.makedirs(results_dir, exist_ok=True)

    ###########
    import time
    time_consumption = 0
    ###########
    for i, data in enumerate(tqdm(test_loader, unit='img'), 0):
        lq_l = data['lq_l'].cuda()
        B, C, H, W = lq_l.shape
        gt_l = data['gt_l'].cuda()
        lq_r = data['lq_r'].cuda()
        gt_r = data['gt_r'].cuda()
        names_l = data['name_l']
        names_r = data['name_r']

        # split data
        split_datas_l, starts_l = splitimage(lq_l, crop_size=opt['datasets']['test']['crop_size'],
                                             overlap_size=opt['datasets']['test']['overlap_size'])
        split_datas_r, starts_r = splitimage(lq_r, crop_size=opt['datasets']['test']['crop_size'],
                                             overlap_size=opt['datasets']['test']['overlap_size'])
        ######
        start_time = time.time()
        ######
        with torch.no_grad():
            for i, data in enumerate(zip(split_datas_l, split_datas_r)):
                _, _, restored_l, restored_r = model(data[0], data[1])
                split_datas_l[i] = restored_l
                split_datas_r[i] = restored_r

                ######
                functional.reset_net(model)
                ######


    # SOPs and param
    step_mode = 's'
    T = opt['network']['T']
    input_size = (3, test_dataset_opt['crop_size'], test_dataset_opt['crop_size'])
    x_l = torch.rand(input_size).cuda().unsqueeze(0)
    x_r = torch.rand(input_size).cuda().unsqueeze(0)
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break

    ops, params = profile(
        model, inputs=(x_l, x_r,),
        verbose=False, custom_ops={
            Conv3x3: count_convNd,
            Conv1x1: count_convNd,
            Linear: count_linear,
            SpikingMatmul: count_matmul,
        })[0:2]
    if step_mode == 'm':
        ops, params = (ops / (1000 ** 3)) / T, params / (1000 ** 2)
    else:
        ops, params = (ops / (1000 ** 3)), params / (1000 ** 2)
    functional.reset_net(model)
    logger.info('mode: {}'.format(step_mode))
    logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        if sublist:
            sop = torch.cat(sublist).mean().item()
            sops = sops + sop
    sops = sops / (1000 ** 3)
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / test_dataset_opt['batch_size']
    if step_mode == 's':
        sops = sops * T
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops)))