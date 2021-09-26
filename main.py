# -*- coding: utf-8 -*-

import os
from model.myNetworks.SARNet_resnet50 import SARNet_r as net
from config.config import cfg

from core.train.trainer_SARNet import Trainer
from core.inference.infer import Inference

model = net().cuda()


def train():
    Trainer(model=model).run()


def infer():
    Inference(mode=1, well_trained_model=model).run()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if cfg.IS_TRAIN:
        train()
    else:
        infer()
