from config.config import cfg
from core.libs import set_logger, mk_dirs_r, Accuracy, init_test_xls
from core.data.data_loader import test_dataset
import os
import torch
import numpy as np
import xlwt
import torch.nn.functional as F
import cv2

logger = set_logger()
test_workbook = xlwt.Workbook()
test_worksheet = init_test_xls(test_workbook.add_sheet('test accuracy', cell_overwrite_ok=False))
acc_score = os.path.join(cfg.TEST.SAVE_DIR1, '{}_{}_infer.xls'.format(cfg.MODEL.NAME, cfg.DATASET.NAME))
acc_score_txt = os.path.join(cfg.TEST.SAVE_DIR1, '{}_{}_infer.txt'.format(cfg.MODEL.NAME, cfg.DATASET.NAME))


def infer_SARNet(model):
    logger.info('Start inference....')

    mk_dirs_r(cfg.TEST.SAVE_DIR1)
    # file_name_list = get_test_im_name(cfg.DATASET.TEST_SET)
    # metric = Accuracy()

    with torch.no_grad():
        image_root = os.path.join(cfg.DATASET.TEST_SET, 'image/')
        gt_root = os.path.join(cfg.DATASET.TEST_SET, 'mask/')
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(test_loader.size):
            image, gt, name, path = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res5, res4, res3, res2, res1, res0 = model(image)
            res = res0
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res = np.uint8(res * 255)
            out = cfg.TEST.SAVE_DIR1 + name[:-4] + '.png'
            pred = np.uint8(res)

            cv2.imwrite(out, pred)
