from collections import OrderedDict

import numpy as np

from data.dataset import create_patch
from evaluation.evaluate_12ECG_score import compute_auc
from utils import AverageMeter
from runner.base import BaseRunner


class ClassificationRunner(BaseRunner):
    def __init__(
        self,
        model,
        dataloader,
        device,
        criterion,
        patch_len,
        stride,
        optimizer=None,
        scheduler=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.patch_len = patch_len
        self.stride = stride

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.model.train()
        return self.forward()

    def evaluate(self, epoch_num=None):
        self.model.eval()
        return self.forward()

    def forward(self, epoch_num=None):
        loss_meter = AverageMeter()
        probs = []
        lbls = []

        for batch in self.dataloader:
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            y_pred = self.model(X)

            loss = self.criterion(y_pred, y)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.item(), n=X.shape[0])
            prob = y_pred.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(y.data.cpu().numpy())

        # scheduler for learning rate change every epoch
        if self.scheduler is not None:
            self.scheduler.step()

        self.epoch_metrics["loss"] = loss_meter.avg
        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)
        auroc, _ = compute_auc(lbls, probs)
        self.epoch_metrics["auroc"] = auroc

        return self.epoch_metrics
