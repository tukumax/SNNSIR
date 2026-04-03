import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from skimage import measure
from torch.nn import init
from utils import read_img, augment, align, hwc_to_chw, get_hr_lr_imgs, get_hr_lr_imgs_mono
import random


class K15_dataset(Dataset):
    """Some Information about K12(K15)_dataset"""

    def __init__(self, root, crop_size=[256, 256], mode='train'):
        super(K15_dataset, self).__init__()
        # ---- root == './' ---
        self.crop_size = crop_size
        self.root = root
        self.mode = mode
        self.gt_images_l = sorted(os.listdir(os.path.join(self.root, 'image_2_3_norain')))
        self.lq_images_l = sorted(os.listdir(os.path.join(self.root, 'image_2_3_rain50')))
        self.gt_images_r = sorted(os.listdir(os.path.join(self.root, 'image_3_2_norain')))
        self.lq_images_r = sorted(os.listdir(os.path.join(self.root, 'image_3_2_rain50')))
        self.img_num = len(self.gt_images_l)
        # print(self.img_num)
        # assert False

    def __getitem__(self, index):
        crop_height, crop_width = self.crop_size
        # gt_img_l = read_img(os.path.join(self.root, 'image_2_3_norain',self.gt_images_l[index]))*2-1
        gt_img_l = read_img(os.path.join(self.root, 'image_2_3_norain', self.gt_images_l[index]))
        lq_img_l = read_img(os.path.join(self.root, 'image_2_3_rain50', self.lq_images_l[index]))
        gt_img_r = read_img(os.path.join(self.root, 'image_3_2_norain', self.gt_images_r[index]))
        lq_img_r = read_img(os.path.join(self.root, 'image_3_2_rain50', self.lq_images_r[index]))

        height, width, _ = lq_img_l.shape
        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_img_l))

        if self.mode == 'train':
            [gt_img_l, lq_img_l] = augment([gt_img_l, lq_img_l], crop_height, crop_width)
            [gt_img_r, lq_img_r] = augment([gt_img_r, lq_img_r], crop_height, crop_width)
        elif self.mode == 'valid':
            [gt_img_l, lq_img_l] = align([gt_img_l, lq_img_l], crop_height, crop_width)
            [gt_img_r, lq_img_r] = align([gt_img_r, lq_img_r], crop_height, crop_width)

        return {'lq_l': hwc_to_chw(lq_img_l), 'gt_l': hwc_to_chw(gt_img_l), 'name_l': self.gt_images_l[index],
                'lq_r': hwc_to_chw(lq_img_r), 'gt_r': hwc_to_chw(gt_img_r), 'name_r': self.gt_images_r[index]}

    def __len__(self):
        return self.img_num


class StereoWaterdrop_dataset(Dataset):
    """Some Information about StereoWaterdrop dataset"""

    def __init__(self, root, crop_size=[256, 256], mode='train'):
        super(StereoWaterdrop_dataset, self).__init__()
        # ---- root == './' ---
        self.crop_size = crop_size
        self.root = root  # train/gt/(left, right), train/input/(left, right)
        self.mode = mode
        self.gt_images_l = sorted(os.listdir(os.path.join(self.root, 'gt', 'left')))
        self.lq_images_l = sorted(os.listdir(os.path.join(self.root, 'input', 'left')))
        self.gt_images_r = sorted(os.listdir(os.path.join(self.root, 'gt', 'right')))
        self.lq_images_r = sorted(os.listdir(os.path.join(self.root, 'input', 'right')))
        self.img_num = len(self.gt_images_l)

    def __getitem__(self, index):
        crop_height, crop_width = self.crop_size
        # gt_img_l = read_img(os.path.join(self.root, 'image_2_3_norain',self.gt_images_l[index]))*2-1
        gt_img_l = read_img(os.path.join(self.root, 'gt', 'left', self.gt_images_l[index]))
        lq_img_l = read_img(os.path.join(self.root, 'input', 'left', self.lq_images_l[index]))
        gt_img_r = read_img(os.path.join(self.root, 'gt', 'right', self.gt_images_r[index]))
        lq_img_r = read_img(os.path.join(self.root, 'input', 'right', self.lq_images_r[index]))

        height, width, _ = lq_img_l.shape
        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_img_l))

        if self.mode == 'train':
            [gt_img_l, lq_img_l] = augment([gt_img_l, lq_img_l], crop_height, crop_width)
            [gt_img_r, lq_img_r] = augment([gt_img_r, lq_img_r], crop_height, crop_width)
        elif self.mode == 'valid':
            [gt_img_l, lq_img_l] = align([gt_img_l, lq_img_l], crop_height, crop_width)
            [gt_img_r, lq_img_r] = align([gt_img_r, lq_img_r], crop_height, crop_width)

        return {'lq_l': hwc_to_chw(lq_img_l), 'gt_l': hwc_to_chw(gt_img_l), 'name_l': self.gt_images_l[index],
                'lq_r': hwc_to_chw(lq_img_r), 'gt_r': hwc_to_chw(gt_img_r), 'name_r': self.gt_images_r[index]}

    def __len__(self):
        return self.img_num


class PairDataset(Dataset):
    def __init__(self, root, crop_size=[256, 256], mode='train', only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

        self.root = root
        self.img_names = sorted(os.listdir(os.path.join(self.root, 'input')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
        crop_height, crop_width = self.crop_size

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        input_img = read_img(os.path.join(self.root, 'input', img_name))
        target_img = read_img(os.path.join(self.root, 'gt', img_name))

        if self.mode == 'train':
            [input_img, target_img] = augment([input_img, target_img], crop_height, crop_width, self.only_h_flip)

        if self.mode == 'valid':
            [input_img, target_img] = align([input_img, target_img], crop_height, crop_width)

        return {'input': hwc_to_chw(input_img), 'gt': hwc_to_chw(target_img), 'filename': img_name}


class StereoSR_dataset(Dataset):
    """Some Information about StereoWaterdrop dataset"""

    def __init__(self, root, crop_size=[30, 90], scale=2, mode='test', vision_mode='bino', left_right='left'):
        super(StereoSR_dataset, self).__init__()

        self.crop_size = crop_size
        self.root = root  # train/gt/(left, right), train/input/(left, right)
        self.mode = mode
        self.scale = scale
        self.vision_mode = vision_mode
        self.left_right = left_right
        if self.mode == 'test':
            self.gt_images_l = sorted(os.listdir(os.path.join(self.root, 'gt', self.left_right)))
            self.lq_images_l = sorted(os.listdir(os.path.join(self.root, 'input', self.left_right)))
            if self.vision_mode == 'bino':
                self.gt_images_r = sorted(os.listdir(os.path.join(self.root, 'gt', 'right')))
                self.lq_images_r = sorted(os.listdir(os.path.join(self.root, 'input', 'right')))
        else:  # train and val
            self.gt_images_l = sorted(os.listdir(os.path.join(self.root, self.left_right)))
            if self.vision_mode == 'bino':
                self.gt_images_r = sorted(os.listdir(os.path.join(self.root, 'right')))
        self.img_num = len(self.gt_images_l)

    def __getitem__(self, index):
        crop_height, crop_width = self.crop_size
        if self.mode == 'test':
            gt_img_l = read_img(os.path.join(self.root, 'gt', self.left_right, self.gt_images_l[index]))
            lq_img_l = read_img(os.path.join(self.root, 'input', self.left_right, self.lq_images_l[index]))
            if self.vision_mode == 'bino':
                gt_img_r = read_img(os.path.join(self.root, 'gt', 'right', self.gt_images_r[index]))
                lq_img_r = read_img(os.path.join(self.root, 'input', 'right', self.lq_images_r[index]))
        else:
            gt_img_l = read_img(os.path.join(self.root, self.left_right, self.gt_images_l[index]))
            if self.vision_mode == 'bino':
                gt_img_r = read_img(os.path.join(self.root, 'right', self.gt_images_r[index]))
                gt_img_l, gt_img_r, lq_img_l, lq_img_r = get_hr_lr_imgs(gt_img_l, gt_img_r, crop_height, crop_width,
                                                                        self.scale)
            else:
                gt_img_l, lq_img_l = get_hr_lr_imgs_mono(gt_img_l, crop_height, crop_width, self.scale)

            if self.mode == 'train':
                # 水平或垂直翻转
                if random.randint(0, 1) == 1:
                    # 水平翻转
                    gt_img_l = np.flip(gt_img_l, axis=1)
                    lq_img_l = np.flip(lq_img_l, axis=1)
                if random.randint(0, 1) == 1:
                    # 垂直翻转
                    gt_img_l = np.flip(gt_img_l, axis=0)
                    lq_img_l = np.flip(lq_img_l, axis=0)

                if self.vision_mode == 'bino':
                    # 水平翻转
                    if random.randint(0, 1) == 1:
                        gt_img_r = np.flip(gt_img_r, axis=1)
                        lq_img_r = np.flip(lq_img_r, axis=1)
                    # 垂直翻转
                    if random.randint(0, 1) == 1:
                        gt_img_r = np.flip(gt_img_r, axis=0)
                        lq_img_r = np.flip(lq_img_r, axis=0)

        if self.vision_mode == 'bino':
            return {'lq_l': hwc_to_chw(lq_img_l), 'gt_l': hwc_to_chw(gt_img_l), 'name_l': self.gt_images_l[index],
                    'lq_r': hwc_to_chw(lq_img_r), 'gt_r': hwc_to_chw(gt_img_r), 'name_r': self.gt_images_r[index]}
        else:
            return {'lq_l': hwc_to_chw(lq_img_l), 'gt_l': hwc_to_chw(gt_img_l), 'name_l': self.gt_images_l[index]}

    def __len__(self):
        return self.img_num


class PairDatasetByPath(Dataset):
    def __init__(self, lq_path, gt_path, crop_size=[256, 256], mode='train', only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

        self.lq_path = lq_path
        self.gt_path = gt_path
        self.lq_images = sorted(os.listdir(lq_path))
        self.gt_images = sorted(os.listdir(gt_path))

        self.img_num = len(self.lq_images)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        crop_height, crop_width = self.crop_size

        # read image, and scale [0, 1] to [-1, 1]
        img_name_lq = self.lq_images[idx]
        img_name_gt = self.gt_images[idx]
        input_img = read_img(os.path.join(self.lq_path, img_name_lq))
        target_img = read_img(os.path.join(self.gt_path, img_name_gt))

        if self.mode == 'train':
            [input_img, target_img] = augment([input_img, target_img], crop_height, crop_width, self.only_h_flip)

        if self.mode == 'valid':
            [input_img, target_img] = align([input_img, target_img], crop_height, crop_width)

        return {'input': hwc_to_chw(input_img), 'gt': hwc_to_chw(target_img), 'filename': img_name_lq}


