from data.dataset import create_patch
from utils import AverageMeter
from runner.base import BaseRunner
import torch


from collections import OrderedDict


class ForecastingRunner(BaseRunner):
    def __init__(
        self,
        model,
        revin,
        dataloader,
        device,
        criterion,
        patch_len,
        stride,
        optimizer=None,
        scheduler=None,
    ):
        self.model = model
        self.revin = revin
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.patch_len = patch_len
        self.stride = stride

        self.l1_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_loss = torch.nn.MSELoss(reduction="mean")

        self.epoch_metrics = OrderedDict()

    def train_epoch(self):
        self.model.train()
        return self.forward()

    def evaluate(self, perturbation_std=None):
        self.model.eval()
        return self.forward(perturbation_std=perturbation_std)

    def forward(self, perturbation_std=None):
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        for batch in self.dataloader:
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            if perturbation_std is not None:
                X = X + torch.randn_like(X) * perturbation_std

            # reversible instance normalization
            if self.revin is not None:
                bs, seq_len, n_vars = X.shape
                X = X.transpose(1, 2).reshape(bs * n_vars, seq_len).unsqueeze(-1)
                X = self.revin(X, "norm")
                X = X.squeeze(-1).reshape(bs, n_vars, seq_len).transpose(1, 2)

            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            y_pred = self.model(X)

            # denormalization
            if self.revin is not None:
                bs, pred_len, n_vars = y_pred.shape
                y_pred = (
                    y_pred.transpose(1, 2).reshape(bs * n_vars, pred_len).unsqueeze(-1)
                )
                y_pred = self.revin(y_pred, "denorm")
                y_pred = (
                    y_pred.squeeze(-1).reshape(bs, n_vars, pred_len).transpose(1, 2)
                )

            loss = self.criterion(y_pred, y)

            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.item(), n=X.shape[0])
            mse = self.l2_loss(y_pred, y)
            mse_meter.update(mse.item(), n=X.shape[0])
            mae = self.l1_loss(y_pred, y)
            mae_meter.update(mae.item(), n=X.shape[0])

        # scheduler for learning rate change every epoch
        if self.scheduler is not None:
            self.scheduler.step()

        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["mse"] = mse_meter.avg
        self.epoch_metrics["mae"] = mae_meter.avg

        return self.epoch_metrics
