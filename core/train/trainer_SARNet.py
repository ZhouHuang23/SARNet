import torch
import os
import time
from config.config import cfg
from core.train.train_libs import print_train_hyper_params, parser_optimizer, parser_lr_schedule, epoch_visualize
from core.data.data_loader import load_train_data, load_val_data
from core.libs import Accuracy, mk_dirs_r, set_logger
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from model.loss_function.train_loss import StructureLoss

torch.manual_seed(cfg.CUDNN.SEED)
torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

logger = set_logger()
scaler = GradScaler()


class Trainer:
    def __init__(self, model):
        print_train_hyper_params(cfg, logger)

        self.model = model.cuda()
        self.optimizer = parser_optimizer(cfg, self.model)
        self.loss_fn = StructureLoss().cuda()

        self.train_loader, self.train_dt_size = load_train_data()
        self.val_loader, self.val_dt_size = load_val_data()

        self.writer = SummaryWriter(cfg.LOG.SAVE_DIR)

        # training initialization
        mk_dirs_r(cfg.LOG.SAVE_DIR)
        mk_dirs_r(cfg.CKPT.SAVE_DIR)

        self.current_epoch = 0
        self.lr_scheduler = parser_lr_schedule(cfg, self.optimizer)

    def __train(self):
        """
            employ the mixed precision strategy to perform train procedure for one epoch
            return: the epoch's training score
        """
        acc = Accuracy()

        self.model.train()
        for itr, (images, gts) in enumerate(self.train_loader):
            images, gts = images.cuda(), gts.cuda()

            # forward
            self.optimizer.zero_grad()  # zero gradient

            with autocast():
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0 = self.model(
                    images)
                loss5 = self.loss_fn(lateral_map_5, gts)
                loss4 = self.loss_fn(lateral_map_4, gts)
                loss3 = self.loss_fn(lateral_map_3, gts)
                loss2 = self.loss_fn(lateral_map_2, gts)
                loss1 = self.loss_fn(lateral_map_1, gts)
                loss0 = self.loss_fn(lateral_map_0, gts)
                loss = (loss5 / 16) + (loss4 / 8) + (loss3 / 4) + (loss2 / 2) + loss1 + loss0

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # 5.Update LR
            if cfg.TRAIN.LR_SCHEDULE.upper() == 'poly'.upper():
                self.lr_scheduler.step()

            # 6.output logs
            batch_log = acc.cal_mini_batch_acc(pred=lateral_map_0, target=gts, loss=loss.item())
            batch_log = "Training epochs:{}/{}, steps:{}/{}, {}".format(self.current_epoch, cfg.TRAIN.EPOCHS, itr + 1,
                                                                        self.train_dt_size, batch_log)
            logger.debug(batch_log)

        return acc.cal_train_epoch_acc()

    def __validate(self):
        acc = Accuracy()

        self.model.eval()
        with torch.no_grad():
            for itr, (images, gts) in enumerate(self.val_loader):
                images, gts = images.cuda(), gts.cuda()

                with autocast():
                    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0 = self.model(
                        images)
                    loss5 = self.loss_fn(lateral_map_5, gts)
                    loss4 = self.loss_fn(lateral_map_4, gts)
                    loss3 = self.loss_fn(lateral_map_3, gts)
                    loss2 = self.loss_fn(lateral_map_2, gts)
                    loss1 = self.loss_fn(lateral_map_1, gts)
                    loss0 = self.loss_fn(lateral_map_0, gts)
                    loss = (loss5 / 16) + (loss4 / 8) + (loss3 / 4) + (loss2 / 2) + loss1 + loss0

                batch_log = acc.cal_mini_batch_acc(pred=lateral_map_0, target=gts, loss=loss.item())
                batch_log = "validate epochs: {}/{}, steps:{}/{}, {}".format(self.current_epoch, cfg.TRAIN.EPOCHS,
                                                                             itr + 1,
                                                                             self.val_dt_size, batch_log)
                logger.debug(batch_log)

        return acc.cal_val_epoch_acc()

    def run(self):
        tic = time.time()  # record training time

        logger.info('Start training...')
        epoch_iou_list = []  # storage IoU value for each epoch

        for epoch in range(cfg.TRAIN.EPOCHS):
            self.current_epoch += 1

            "training one epoch"
            # ------------------------------------------------------------------------------------------
            start_time = time.time()  # record the start time for each epoch
            train_score, train_logs = self.__train()
            end_time = time.time()  # record the end time for each epoch

            # generate logs
            train_score.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_train_time = (end_time - start_time) / 60
            train_score.append(epoch_train_time)
            train_logs = "Training epochs:{}/{}, {}, elapsed time:{:.2f} min".format(self.current_epoch,
                                                                                     cfg.TRAIN.EPOCHS,
                                                                                     train_logs,
                                                                                     epoch_train_time)
            # ------------------------------------------------------------------------------------------

            "Validate one epoch"
            # ------------------------------------------------------------------------------------------
            start_time = time.time()  # record the start time for each epoch
            val_score, val_logs = self.__validate()
            end_time = time.time()  # record the end time for each epoch

            # generate logs
            epoch_val_time = (end_time - start_time) / 60
            val_score.append(epoch_val_time)
            val_logs = "Validate epochs:{}/{}, {}, elapsed time:{:.2f} min".format(self.current_epoch, cfg.TRAIN.EPOCHS,
                                                                                   val_logs, epoch_val_time)
            # ------------------------------------------------------------------------------------------

            """display and save training and validation logs"""
            logger.debug('--' * 60)
            logger.warning('Epoch training-validation logs:')
            logger.debug(train_logs)
            logger.debug(val_logs)
            logger.debug('--' * 60)

            epoch_visualize(curr_epoch=self.current_epoch, writer=self.writer, train_score=train_score,
                            val_score=val_score)

            # save networks CKPT.SAVE_DIR
            if self.current_epoch >= cfg.CKPT.NUM:
                epoch_iou_list.append(val_score[5])
                weight_path = os.path.join(cfg.CKPT.SAVE_DIR, '%d-ckpt.pth' % self.current_epoch)
                all_states = {"net": self.model.state_dict(), cfg.TRAIN.OPTIMIZER: self.optimizer.state_dict(),
                              "epoch": epoch}
                torch.save(obj=all_states, f=weight_path)

                # adjust learning rate after each epoch complete train-eval procedure
                if cfg.TRAIN.LR_SCHEDULE.upper() != 'poly'.upper():
                    self.lr_scheduler.step()

        toc = time.time()
        logger.error('<<<<<<<<<<<<<< End Training >>>>>>>>>>>>>>')
        logger.info('Total elapsed time is {:.2f} hours.'.format((toc - tic) / 3600))

        optima_epoch = epoch_iou_list.index(max(epoch_iou_list))
        optima_epoch += cfg.CKPT.NUM
        logger.info("The best epoch is Epoch:{}, best IoU={}".format(optima_epoch, max(epoch_iou_list)))
