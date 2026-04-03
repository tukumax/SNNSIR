from utils import load_yaml_config, save_yaml_config, PerceptualLoss, \
    generate_timestamp, K15_dataset, StereoWaterdrop_dataset, torchPSNR, torchSSIM, mkdir, get_logger
from models import create_model
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from thop import profile
import numpy as np
import os
from natsort import natsorted
from glob import glob
import random
import time
from spikingjelly.activation_based import functional
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./configs/unet5_wwLifMul_1CAM_snn_waterdrop_32to160_vth02_SRBB_SSCM_SSCA_SSRB_1e3_4_c32.yml' , help='Path to option YAML file.')
    args = parser.parse_args()
    opt = load_yaml_config(args.opt)

    return opt, args.opt


def train(opt, yaml_path):


    ### gup num
    device = str(opt['datasets']['train']['gpu_id'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    # 设置设备
    # torch.cuda.set_device(rank)

    ### Set Seeds
    random.seed(opt['manual_seed'])
    np.random.seed(opt['manual_seed'])
    torch.manual_seed(opt['manual_seed'])
    torch.cuda.manual_seed_all(opt['manual_seed'])

    ### model
    model_save_dir = os.path.join(opt['train']['save_dir'], opt['net_name'],
                                  generate_timestamp(generate_format="%Y-%m-%d-%H-%M-%S"))
    mkdir(model_save_dir)
    logger = get_logger(opt['model_name'], os.path.join(model_save_dir, opt['model_name']+'_log.log'))

    model = create_model(opt)
    logger.info(model)
    model.cuda()
    if len(device)>1:
        model = torch.nn.DataParallel(model)

    ### loss
    l1_loss = torch.nn.L1Loss().cuda()
    perceptual_loss = PerceptualLoss().cuda()

    ### scheduler
    train_opt = opt['train']
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            logger.info(f'Params {k} will not be optimized.')

    optim_type = train_opt['optim'].pop('kind')
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(optim_params, **train_opt['optim'])
        train_opt['optim']['kind'] = 'Adam'
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(optim_params, **train_opt['optim'])
        train_opt['optim']['kind'] = 'AdamW'
    else:
        raise logger.error(NotImplementedError(
            f'optimizer {optim_type} is not supperted yet.'))
    # optimizer = torch.optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True], lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler_type = train_opt['scheduler'].pop('kind')
    if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
        scheduler = torch.optim.lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler'])
        train_opt['scheduler']['kind'] = 'MultiStepRestartLR'
    elif scheduler_type == 'CosineAnnealingRestartLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler'])
        train_opt['scheduler']['kind'] = 'CosineAnnealingRestartLR'
    elif scheduler_type == 'CosineAnnealingWarmupRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmupRestarts(optimizer, **train_opt['scheduler'])
        train_opt['scheduler']['kind'] = 'CosineAnnealingWarmupRestarts'
    elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler'])
        train_opt['scheduler']['kind'] = 'CosineAnnealingRestartCyclicLR'
    elif scheduler_type == 'CosineAnnealingLR':
        print('..', 'cosineannealingLR')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler'])
        train_opt['scheduler']['kind'] = 'CosineAnnealingLR'
    elif scheduler_type == 'CosineAnnealingLRWithRestart':
        print('..', 'CosineAnnealingLR_With_Restart')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler'])
        train_opt['scheduler']['kind'] = 'CosineAnnealingLRWithRestart'
    elif scheduler_type == 'LinearLR':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, train_opt['total_iter'])
        train_opt['scheduler']['kind'] = 'LinearLR'
    elif scheduler_type == 'VibrateLR':
        scheduler = torch.optim.lr_scheduler.VibrateLR(optimizer, train_opt['total_iter'])
        train_opt['scheduler']['kind'] = 'VibrateLR'
    else:
        raise logger.error(NotImplementedError(
            f'Scheduler {scheduler_type} is not implemented yet.'))


    start_epoch = 1
    ### resume
    if opt['resume']:
        path_chk_rest = natsorted(glob(os.path.join(opt['path']['resume_state'], '*%s' % '_last.pth')))[-1]
        checkpoint = torch.load(path_chk_rest)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_last_lr()[0]
        logger.info('------------------------------------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:", new_lr)
        logger.info('------------------------------------------------------------------------------')

    ### dataloader
    dataset_opt = opt['datasets']
    if dataset_opt['kind'] == 'k15':
        dataset_train = K15_dataset(dataset_opt['train']['dataroot'], dataset_opt['train']['patch_size'], 'train')
        train_loader = DataLoader(dataset=dataset_train, num_workers=dataset_opt['train']['num_worker_per_gpu'],
                                  batch_size=dataset_opt['train']['batch_size'],
                                  shuffle=dataset_opt['train']['use_shuffle'],
                                  drop_last=False)
        dataset_val = K15_dataset(dataset_opt['val']['dataroot'], dataset_opt['val']['patch_size'], 'valid')
        val_loader = DataLoader(dataset=dataset_val, num_workers=dataset_opt['val']['num_worker_per_gpu'],
                                batch_size=dataset_opt['val']['batch_size'],
                                shuffle=False, drop_last=False)
    elif dataset_opt['kind'] == 'StereoWaterdrop':
        dataset_train = StereoWaterdrop_dataset(dataset_opt['train']['dataroot'], dataset_opt['train']['patch_size'],
                                                'train')
        train_loader = DataLoader(dataset=dataset_train, num_workers=dataset_opt['train']['num_worker_per_gpu'],
                                  batch_size=dataset_opt['train']['batch_size'],
                                  shuffle=dataset_opt['train']['use_shuffle'],
                                  drop_last=False)
        dataset_val = StereoWaterdrop_dataset(dataset_opt['val']['dataroot'], dataset_opt['val']['patch_size'], 'valid')
        val_loader = DataLoader(dataset=dataset_val, num_workers=dataset_opt['val']['num_worker_per_gpu'],
                                batch_size=dataset_opt['val']['batch_size'],
                                shuffle=False, drop_last=False)

    ### start training
    logger.info('===> Start Epoch {} End Epoch {}'.format(start_epoch, train_opt['epochs']))
    logger.info('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    writer = SummaryWriter(model_save_dir)
    scaler = torch.amp.GradScaler('cuda')
    iter = 0

    start_time = time.time()
    for epoch in range(start_epoch, train_opt['epochs'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_psnr_rgb = []
        train_ssim_rgb = []
        model.train()
        for i, data in enumerate(train_loader, 0):
            lq_l = data['lq_l'].cuda()
            gt_l = data['gt_l'].cuda()
            lq_r = data['lq_r'].cuda()
            gt_r = data['gt_r'].cuda()

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                restored_l, restored_r, refined_l, refined_r = model(lq_l, lq_r)
                loss_l = l1_loss(restored_l, gt_l) + l1_loss(restored_r, gt_r)
                loss_p_refine = perceptual_loss(refined_l, gt_l, batch_size=dataset_opt['train']['batch_size']) + \
                                perceptual_loss(refined_r, gt_r, batch_size=dataset_opt['train']['batch_size'])
                loss = loss_p_refine + loss_l

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            iter += 1
            ######
            functional.reset_net(model)
            ######

            for res_l, tar_l, res_r, tar_r in zip(refined_l, gt_l, refined_r, gt_r):
                train_psnr_rgb.append((torchPSNR(res_l, tar_l)+torchPSNR(res_r, tar_r))/2)
            train_ssim_rgb.append((torchSSIM(refined_l, gt_l)+torchSSIM(refined_r, gt_r))/2)
            writer.add_scalar('loss/iter_loss', loss.item(), iter)

        psnr_train = torch.stack(train_psnr_rgb).mean().item()
        ssim_train = torch.stack(train_ssim_rgb).mean().item()
        writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
        writer.add_scalar('lr/epoch_loss', scheduler.get_last_lr()[0], epoch)

        ### evaluation
        if epoch % dataset_opt['val']['val_epochs'] == 0:
            model.eval()
            val_psnr_rgb = []
            for i, data in enumerate(val_loader, 0):
                lq_l = data['lq_l'].cuda()
                gt_l = data['gt_l'].cuda()
                lq_r = data['lq_r'].cuda()
                gt_r = data['gt_r'].cuda()

                with torch.no_grad():
                    # restored_l, restored_r = model(lq_l, lq_r)
                    _, _, refined_l, refined_r = model(lq_l, lq_r)

                ######
                functional.reset_net(model)
                ######

                for res_l, tar_l, res_r, tar_r in zip(refined_l, gt_l, refined_r, gt_r):
                    val_psnr_rgb.append((torchPSNR(res_l, tar_l) + torchPSNR(res_r, tar_r)) / 2)
            psnr_val = torch.stack(val_psnr_rgb).mean().item()
            writer.add_scalar('val/psnr', psnr_val, epoch)

            ### save best model
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(model_save_dir, "model_best.pth"))
            logger.info("[epoch %d Training PSNR: %.4f --- best_epoch %d Test_PSNR %.4f]" % (epoch, psnr_train, best_epoch, best_psnr))

        ### save last model
        if epoch % train_opt['save_freq'] == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_save_dir, f"model_epoch_{epoch}.pth"))
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_save_dir, "model_last.pth"))
        scheduler.step()
        logger.info("-" * 150)
        logger.info(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tTrain_PSNR: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}\tTest_PSNR: {:.4f}".format(
                epoch, time.time() - epoch_start_time, loss.item(), psnr_train, ssim_train, scheduler.get_last_lr()[0],
                best_psnr, ))
        logger.info("-" * 150)
    writer.close()

    logger.info('Total time:{:.2f}h'.format((time.time()-start_time) / 60 ** 2))

    opt['path']['saved_path'] = model_save_dir
    logger.info("#######################")
    logger.info(opt)
    logger.info(model_save_dir)
    logger.info("#######################")
    save_yaml_config(opt, yaml_path)

    ### copy model file and yml file to model_save_dir
    # shutil.copy2(source_path, destination_path)
    shutil.copy2(os.path.join('./models', opt['model_name']+'.py'), os.path.join(model_save_dir, opt['model_name']+'.py'))
    shutil.copy2(yaml_path, os.path.join(model_save_dir, os.path.basename(yaml_path)))


if __name__ == '__main__':
    opt, yaml_path = parse_args()
    train(opt, yaml_path)

