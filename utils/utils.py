import os
import cv2
import numpy as np
import random
import yaml
from collections import OrderedDict
import time
import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Tuple, Union
from utils.layers import SpikingMatmul
from einops import rearrange
import logging

### tool for super-resolution
# 定义 modcrop 函数
def modcrop(img, scale):
    h = img.shape[0]
    w = img.shape[1]
    img_cropped = img[:int(np.floor(h / scale)) * scale, :int(np.floor(w / scale)) * scale,:]
    return img_cropped

def get_hr_lr_imgs(img_l,img_r, crop_height=30, crop_width=90, scale=2):
    img_hr_l = modcrop(img_l, scale)
    img_hr_r = modcrop(img_r, scale)
    img_lr_l = cv2.resize(img_hr_l, (img_hr_l.shape[1] // scale, img_hr_l.shape[0] // scale),
                          interpolation=cv2.INTER_AREA)
    img_lr_r = cv2.resize(img_hr_r, (img_hr_r.shape[1] // scale, img_hr_r.shape[0] // scale),
                          interpolation=cv2.INTER_AREA)

    H, W, _ = img_lr_l.shape
    Hc, Wc = [crop_height, crop_width]

    H_lr = random.randint(2, H - Hc-3)  # not border
    W_lr = random.randint(2, W - Wc-3)
    H_hr = H_lr * scale
    W_hr = W_lr * scale

    return img_hr_l[H_hr:(H_lr+Hc)*scale, W_hr:(W_lr+Wc)*scale,:], \
           img_hr_r[H_hr:(H_lr+Hc)*scale, W_hr:(W_lr+Wc)*scale,:], \
           img_lr_l[H_lr:(H_lr+Hc),W_lr:(W_lr+Wc),:], img_lr_r[H_lr:(H_lr+Hc),W_lr:(W_lr+Wc),:]

def get_hr_lr_imgs_mono(img_l, crop_height=30, crop_width=90, scale=2):
    img_hr_l = modcrop(img_l, scale)
    img_lr_l = cv2.resize(img_hr_l, (img_hr_l.shape[1] // scale, img_hr_l.shape[0] // scale),
                          interpolation=cv2.INTER_AREA)

    H, W, _ = img_lr_l.shape
    Hc, Wc = [crop_height, crop_width]

    H_lr = random.randint(2, H - Hc-3)  # not border
    W_lr = random.randint(2, W - Wc-3)
    H_hr = H_lr * scale
    W_hr = W_lr * scale

    return img_hr_l[H_hr:(H_lr+Hc)*scale, W_hr:(W_lr+Wc)*scale,:], img_lr_l[H_lr:(H_lr+Hc),W_lr:(W_lr+Wc),:]


### general ###
def generate_timestamp(generate_format="%Y-%m-%d-%H-%M-%S"):
    timestamp = time.time()
    local_time_tuple = time.localtime(timestamp)
    formatted_time = time.strftime(generate_format, local_time_tuple)
    return formatted_time

def get_logger(loggername, log_path):
    # loggername
    logger = logging.getLogger(loggername)
    # log level can be recorded
    logger.setLevel(logging.DEBUG)

    # log file
    file_handler = logging.FileHandler(log_path)

    formatter = logging.Formatter('[%(levelname)s]----%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


### for image process
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def augment(imgs=[], crop_height=64, crop_width=64, only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [crop_height, crop_width]

    Hs = random.randint(0, H - Hc)
    Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        # 对于双目任务，应该采用对称裁剪，保住视差信息
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], crop_height=64, crop_width=64):
    H, W, _ = imgs[0].shape
    Hc, Wc = [crop_height, crop_width]

    # Hs = (H - Hc) // 2
    # Ws = (W - Wc) // 2
    Hs = random.randint(0, H - Hc)
    Ws = random.randint(0, W - Wc)
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


def read_img(filename):
    img = cv2.imread(filename)
    return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
    img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    cv2.imwrite(filename, img)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def torchPSNR(prd_img, tar_img):
    '''by rgb'''
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def torchSSIM(img1, img2, window_size=11, size_average=True):
    # img1,img2 torch.tensor
    # B C H W
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


### for yaml file ###
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    def str_representer(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Dumper.add_representer(list, list_representer)
    Dumper.add_representer(str, str_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def load_yaml_config(yaml_path):
    with open(yaml_path, mode='r', encoding='utf-8') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    return opt


def save_yaml_config(opt, yaml_path):
    with open(yaml_path, 'w') as file:
        _, Dumper = ordered_yaml()
        yaml.dump(opt, file, Dumper=Dumper, default_flow_style=False)


#### for yaml file ####

###
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

### params and energy
def unpack_for_conv(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1) if x.dim()>4 else x # t b -> t*b

def unpack_for_conv1d(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1) if x.shape[0]<=8 and x.dim()>3 else x # t b -> t*b


def unpack_for_linear(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1) if x.shape[0]<=8 else x  # t b -> t*b


def unpack_for_matmul(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor]:
    assert isinstance(x, tuple)
    assert x.__len__() == 2
    left, right = x
    return left.flatten(0, 1) if left.dim()>4 else left, right.flatten(0, 1) if right.dim()>4 else right # t b -> t*b


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            # if name in net.skip:  #type:ignore
            #     continue
            if any(keyword in name for keyword in net.skip):
                continue  # 跳过包含关键词的模块
            if isinstance(m, nn.Conv2d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_conv(name)))  # type:ignore
            elif isinstance(m, nn.Conv1d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_conv1d(name)))  # type:ignore
            elif isinstance(m, nn.Linear):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_linear(name)))  # type:ignore
            elif isinstance(m, SpikingMatmul):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_matmul(name)))  # type:ignore

    def cal_sop_conv(self, x: Tensor, m: nn.Conv2d):
        with torch.no_grad():
            out = torch.nn.functional.conv2d(x, torch.ones_like(m.weight), None, m.stride,
                                             m.padding, m.dilation, m.groups)
            return out.sum().unsqueeze(0)

    def create_hook_conv(self, name):
        def hook(m, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_conv(unpack_for_conv(x).detach(), m))

        return hook

    def cal_sop_conv1d(self, x: Tensor, m: nn.Conv1d):
        with torch.no_grad():
            out = torch.nn.functional.conv1d(x, torch.ones_like(m.weight), None, m.stride,
                                             m.padding, m.dilation, m.groups)
            return out.sum().unsqueeze(0)

    def create_hook_conv1d(self, name):
        def hook(m, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_conv1d(unpack_for_conv1d(x).detach(), m))

        return hook

    def cal_sop_linear(self, x: Tensor, m: nn.Linear):
        with torch.no_grad():
            out = torch.nn.functional.linear(x, torch.ones_like(m.weight), None)
            return out.sum().unsqueeze(0)

    def create_hook_linear(self, name):
        def hook(m, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_linear(unpack_for_linear(x).detach(), m))

        return hook

    def cal_sop_matmul(self, left: Tensor, right: Tensor, m: SpikingMatmul):
        with torch.no_grad():
            # m参数指的是0-1矩阵的位置
            if m.spike == 'l': right = torch.ones_like(right)
            elif m.spike == 'r': left = torch.ones_like(left)
            elif m.spike == 'both': pass
            else: raise ValueError(m.spike)
            out = torch.matmul(left, right)
            return out.sum().unsqueeze(0)

    def create_hook_matmul(self, name):
        def hook(m, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                left, right = unpack_for_matmul(x)
                self.records.append(self.cal_sop_matmul(left.detach(), right.detach(), m))

        return hook


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int):
    # T, N, out_c, oh, ow = output_size
    # T, N, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[2]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def count_convNd(m, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(input_size=list(x.shape), output_size=list(y.shape),
                                          kernel_size=list(m.weight.shape), groups=m.groups)


def count_matmul(m, x, y):
    # x: input tensor
    # y: output tensor
    left, right = x
    # per output element
    total_mul = right.shape[-1]
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = left.numel()
    # exactly, it's MACs, not FLOPs
    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])

# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])

def calculate_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    # TODO: add test cases
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2
    m.total_ops += flops


def calculate_adaptive_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])

def count_adap_avgpool(m, x, y):
    kernel = torch.div(
        torch.DoubleTensor([*(x[0].shape[2:])]),
        torch.DoubleTensor([*(y.shape[2:])])
    )
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    m.total_ops += calculate_adaptive_avg(total_add, num_elements)

