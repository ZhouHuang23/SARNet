import os
from config.config import cfg
import torch
from core.libs.logger import set_logger
from core.inference.infer_SARNet import infer_SARNet


logger = set_logger()


class Inference:
    def __init__(self, well_trained_model, mode=0):
        """
        @param well_trained_model: trained model
        @param ckpt: checkpoint
        @param mode: mode=1: general prediction, mode=2: dilated prediction
        """
        self.model = well_trained_model
        weight_path = os.path.join(cfg.CKPT.SAVE_DIR, cfg.CKPT.SELECTED_INFER_CKPT)

        self.model.load_state_dict(torch.load(weight_path)['net'])
        self.model.eval()
        logger.info('Model {} checkpoint weight has been loaded successfully...'.format(cfg.MODEL.NAME))

        self.mode = mode

    def run(self):
        if self.mode == 1:
            logger.warning(">>>> Start inference using general prediction manner .")
            infer_SARNet(self.model)




