# -*- coding: utf-8 -*-

from config.config import cfg
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as tf

random.seed(cfg.CUDNN.SEED)
np.random.seed(cfg.CUDNN.SEED)


def random_rotation(image, mask, angle=None, p=1):
    if angle is None:
        angle = [-10, 10]
    if random.random() <= p:
        r = 0
        if isinstance(angle, list):
            r = random.randrange(angle[0], angle[1])
        else:
            assert "angle should be list type, please check the type..."
        image = image.rotate(r)
        mask = mask.rotate(r)

    return image, mask


def random_flip(image, mask, p=1):
    if random.random() <= p:
        if random.random() <= 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        else:
            image = tf.vflip(image)
            mask = tf.vflip(mask)

    return image, mask


def random_resize(image, mask, scale=[0.5, 2], p=1):  # scale表示随机crop出来的图片会在的0.5倍至2倍之间，ratio表示长宽比
    if random.random() <= p:
        rows, cols = image.size[0], image.size[1]
        r = random.randint(scale[0] * 10, scale[1] * 10) / 10

        new_rows, new_cols = int(r * rows), int(r * cols)

        image = tf.resize(image, (new_rows, new_cols), Image.BILINEAR)  # image resize
        mask = tf.resize(mask, (new_rows, new_cols), Image.NEAREST)

        if new_rows > rows:  # resize后的图像尺寸大于原图则crop至原图大小
            image = tf.center_crop(image, output_size=(rows, cols))
            mask = tf.center_crop(mask, output_size=(rows, cols))

        if new_rows < rows:  # resize后的图像尺寸小于原图则pad至原图大小
            padding = int((rows - new_rows) / 2)
            image = tf.pad(image, padding=padding, fill=0, padding_mode='constant')
            mask = tf.pad(mask, padding=padding, fill=0, padding_mode='constant')
            if padding * 2 + new_rows != rows:
                image = tf.resize(image, size=rows)
                mask = tf.resize(mask, size=rows)

    return image, mask


def adjust_contrast(image, mask, scale=0.5, p=1):
    if random.random() <= p:
        image = tf.adjust_contrast(image, scale)
    return image, mask


def adjust_brightness(image, mask, factor=0.125, p=1):
    if random.random() <= p:
        image = tf.adjust_brightness(image, factor)
    return image, mask


def adjust_saturation(image, mask, factor=0.5, p=1):
    if random.random() <= p:
        image = tf.adjust_saturation(image, factor)
    return image, mask


def adjust_hue(image, mask, factor=0.2, p=1):
    if random.random() <= p:
        image = tf.adjust_hue(image, hue_factor=factor)
    return image, mask


def center_crop(image, mask, scale=1, p=1):
    if random.random() <= p:
        rows, cols = image.size[0], image.size[1]
        new_rows = int(rows * scale)
        new_cols = int(cols * scale)
        image = tf.center_crop(image, output_size=(new_rows, new_cols))
        mask = tf.center_crop(mask, output_size=(new_rows, new_cols))

        new_rows, new_cols = image.size[0], image.size[1]

        padding = int((rows - new_rows) / 2)

        image = transforms.Pad(padding=padding, fill=0, padding_mode='constant')(image)
        mask = transforms.Pad(padding=padding, fill=0, padding_mode='constant')(mask)

        if padding * 2 + new_rows != rows:
            image = tf.resize(image, size=rows)
            mask = tf.resize(mask, size=rows)

    return image, mask


def gaussian_blur(image, label, radius=3, p=1):
    if random.random() <= p:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    return image, label


def add_gaussian_noise(image, label, noise_sigma=25, p=1):
    if random.random() <= p:
        temp_image = np.float64(np.copy(image))
        h, w, _ = temp_image.shape
        # 标准正态分布*noise_sigma
        noise = np.random.randn(h, w) * noise_sigma
        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

            image = Image.fromarray(np.uint8(noisy_image))

    return image, label


total_strategy = ['add_gaussian_noise', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'adjust_saturation',
                  'center_crop', 'gaussian_blur', 'random_flip', 'random_resize', 'random_rotation']


def transform(aug_method, im, gt):
    if aug_method == 'random_resize':
        im, gt = random_resize(im, gt)
    elif aug_method == 'random_rotation':
        im, gt = random_rotation(im, gt, angle=[-90, 90])
    elif aug_method == 'random_flip':
        im, gt = random_flip(im, gt)
    elif aug_method == 'gaussian_blur':
        im, gt = gaussian_blur(im, gt, radius=3)
    elif aug_method == 'gaussian_noise':
        im, gt = add_gaussian_noise(im, gt, noise_sigma=25)
    elif aug_method == 'adjust_brightness':
        im, gt = adjust_brightness(im, gt)
    elif aug_method == 'adjust_contrast':
        im, gt = adjust_contrast(im, gt)
    elif aug_method == 'adjust_hue':
        im, gt = adjust_hue(im, gt)
    elif aug_method == 'adjust_saturation':
        im, gt = adjust_saturation(im, gt)
    elif aug_method == 'center_crop':
        im, gt = center_crop(im, gt)
    return im, gt
