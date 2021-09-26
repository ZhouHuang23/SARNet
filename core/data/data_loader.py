# -*- coding: utf-8 -*-

from config.config import cfg
from core.libs.logger import set_logger

import os
import scipy.io as io
from PIL import Image

from torchvision.transforms import transforms as tf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import math
from core.data import transform
import numpy as np

random.seed(cfg.CUDNN.SEED)
np.random.seed(cfg.CUDNN.SEED)

logger = set_logger()


def _make_dataset(im_dir, gt_dir):
    imgs_image = []
    imgs_mask = []
    for f in os.listdir(im_dir):
        if f.endswith(cfg.DATASET.FORMAT_TRAIN_SET):
            imgs_image.append(os.path.join(im_dir, f))
    for f in os.listdir(gt_dir):
        if f.endswith(cfg.DATASET.FORMAT_MASK):
            imgs_mask.append(os.path.join(gt_dir, f))

    return imgs_image, imgs_mask


class MyDataSet(Dataset):
    def __init__(self, dataset_dir, aug_is_need=True, p=None):
        im_dir = os.path.join(dataset_dir, "image")
        gt_dir = os.path.join(dataset_dir, "mask")

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.train_img_size = cfg.TRAIN.TRAIN_IMG_SIZE

        self.aug_is_need = aug_is_need
        self.p = p
        if aug_is_need:
            if len(cfg.TRAIN.AUG_P) != len(cfg.TRAIN.AUG_STRATEGY):
                assert len(cfg.TRAIN.AUG_P) != len(cfg.TRAIN.AUG_STRATEGY)
            if sum(cfg.TRAIN.AUG_P) != 1:
                assert sum(cfg.TRAIN.AUG_P) != 1

        self.x_transform = tf.Compose([
            tf.Resize((self.train_img_size, self.train_img_size)),
            tf.ToTensor(),
            tf.Normalize(mean=mean, std=std)
        ])

        self.y_transform = tf.Compose([
            tf.Resize((self.train_img_size, self.train_img_size)),
            tf.ToTensor()
        ])

        self.im_list, self.gt_list = _make_dataset(im_dir=im_dir, gt_dir=gt_dir)

    def __getitem__(self, index):
        # load image
        im_filepath = self.im_list[index]
        gt_filepath = self.gt_list[index]
        im = Image.open(im_filepath).convert('RGB')
        gt = Image.open(gt_filepath).convert('L')

        if im is None or gt is None:
            print(im_filepath)

        # augmentation
        if self.aug_is_need:
            if random.random() <= self.p:
                strategy = cfg.TRAIN.AUG_STRATEGY
                p = np.array(cfg.TRAIN.AUG_P)
                aug_method = np.random.choice(strategy, p=p.ravel())
                try:
                    im, gt = transform.transform(aug_method=aug_method, im=im, gt=gt)
                except TypeError:
                    print(im_filepath)

        im = self.x_transform(im)
        gt = self.y_transform(gt)

        return im, gt

    def __len__(self):
        return len(self.im_list)


def load_train_data():
    dataset = MyDataSet(dataset_dir=cfg.DATASET.TRAIN_SET, p=cfg.TRAIN.DEFAULT_AUG_P)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    dt_size = math.floor(len(dataloader.dataset) / cfg.TRAIN.BATCH_SIZE)

    logger.warning('Training set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.TRAIN_SET))
    return dataloader, dt_size


def load_val_data():
    dataset = MyDataSet(dataset_dir=cfg.DATASET.VAL_SET, aug_is_need=False)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    dt_size = math.floor(len(dataloader.dataset) / cfg.TRAIN.BATCH_SIZE)
    logger.warning('Validation set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.VAL_SET))
    return dataloader, dt_size


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = tf.Compose([
            tf.Resize((self.testsize, self.testsize)),
            tf.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            tf.Normalize([0.3476, 0.3688, 0.3373], [0.1272, 0.1144, 0.1075])  # ORSSD
        ])
        self.gt_transform = tf.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        path = self.images[self.index]
        image = self.rgb_loader(path)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, gt, name, path

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
