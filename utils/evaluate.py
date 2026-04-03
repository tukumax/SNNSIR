import numpy as np
import os
import cv2
import math
import lpips

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong rst image dimensions.')


def calculate_ms_ssim(img1, img2):
    # range:[0,255.0]
    # 转换为灰度图像
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算MS-SSIM
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])  # 不同尺度的权重
    levels = weights.size

    mssim = np.zeros(levels)
    mcs = np.zeros(levels)

    for i in range(levels):
        ssim_map_mean, cs_map = ssim(img1, img2, mode='ms-ssim')
        # mssim[i] = np.mean(ssim_map)
        mssim[i] = ssim_map_mean
        mcs[i] = np.mean(cs_map)

        img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # 整体MS-SSIM计算
    overall_mssim = np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])

    return overall_mssim


def calculate_lpips(img1_path, img2_path):
    loss_fn = lpips.LPIPS(net='alex', version=0.1)
    img1 = lpips.im2tensor(lpips.load_image(img1_path))
    img2 = lpips.im2tensor(lpips.load_image(img2_path))
    return loss_fn.forward(img1, img2).item()


def ssim(img1, img2, mode='ssim'):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    if mode == 'ssim':
        return ssim_map.mean()
    else:
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return ssim_map.mean(), cs_map


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def evaluate_psnr_ssim(gt_dir, rst_dir):
    # 获取文件夹下的所有文件名（不包含路径）
    gt_names = sorted(os.listdir(gt_dir))
    rst_names = sorted(os.listdir(rst_dir))

    PSNRs, SSIMs = [], []
    # for i in range(0, len(file_names)):
    for gt_name, rst_name in zip(gt_names, rst_names):
        img1 = load_img(os.path.join(gt_dir, gt_name))
        img2 = load_img(os.path.join(rst_dir, rst_name))
        PSNRs.append(round(calculate_psnr(img1, img2), 4))
        SSIMs.append(round(calculate_ssim(img1, img2), 4))
        save_img(os.path.join(gt_dir, gt_name), img1)
        save_img(os.path.join(rst_dir, rst_name), img2)
    # print("[PSNR] mean: {:.4f} std: {:.4f}".format(np.mean(PSNRs), np.std(PSNRs, ddof=1)))
    # print("[SSIM] mean: {:.4f} std: {:.4f}".format(np.mean(SSIMs), np.std(SSIMs, ddof=1)))
    return PSNRs, SSIMs, gt_names, rst_names


def evaluate_psnr_ssim_ms_ssim(gt_dir, rst_dir):
    # 获取文件夹下的所有文件名（不包含路径）
    gt_names = sorted(os.listdir(gt_dir))
    rst_names = sorted(os.listdir(rst_dir))

    PSNRs, SSIMs, MS_SSIMs = [], [], []
    # for i in range(0, len(file_names)):
    for gt_name, rst_name in zip(gt_names, rst_names):
        img1 = load_img(os.path.join(gt_dir, gt_name))
        img2 = load_img(os.path.join(rst_dir, rst_name))
        PSNRs.append(round(calculate_psnr(img1, img2), 4))
        SSIMs.append(round(calculate_ssim(img1, img2), 4))
        MS_SSIMs.append(round(calculate_ms_ssim(img1, img2), 4))
        save_img(os.path.join(gt_dir, gt_name), img1)
        save_img(os.path.join(rst_dir, rst_name), img2)
    # print("[PSNR] mean: {:.4f} std: {:.4f}".format(np.mean(PSNRs), np.std(PSNRs, ddof=1)))
    # print("[SSIM] mean: {:.4f} std: {:.4f}".format(np.mean(SSIMs), np.std(SSIMs, ddof=1)))
    return PSNRs, SSIMs, MS_SSIMs, gt_names, rst_names


def evaluate_psnr_ssim_ms_ssim_lpips(gt_dir, rst_dir):
    # 获取文件夹下的所有文件名（不包含路径）
    gt_names = sorted(os.listdir(gt_dir))
    rst_names = sorted(os.listdir(rst_dir))

    PSNRs, SSIMs, MS_SSIMs = [], [], []
    LPIPSs = []
    # for i in range(0, len(file_names)):
    for gt_name, rst_name in zip(gt_names, rst_names):
        img1 = load_img(os.path.join(gt_dir, gt_name))
        img2 = load_img(os.path.join(rst_dir, rst_name))
        PSNRs.append(round(calculate_psnr(img1, img2), 4))
        SSIMs.append(round(calculate_ssim(img1, img2), 4))
        MS_SSIMs.append(round(calculate_ms_ssim(img1, img2), 4))
        LPIPSs.append(round(calculate_lpips(os.path.join(gt_dir, gt_name), os.path.join(rst_dir, rst_name)), 4))
        save_img(os.path.join(gt_dir, gt_name), img1)
        save_img(os.path.join(rst_dir, rst_name), img2)
    # print("[PSNR] mean: {:.4f} std: {:.4f}".format(np.mean(PSNRs), np.std(PSNRs, ddof=1)))
    # print("[SSIM] mean: {:.4f} std: {:.4f}".format(np.mean(SSIMs), np.std(SSIMs, ddof=1)))
    return PSNRs, SSIMs, MS_SSIMs, LPIPSs, gt_names, rst_names