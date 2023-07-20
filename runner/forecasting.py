# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import torch
from data.mixup import mixup_criterion, mixup_data
from runner.base import BaseRunner

logger = logging.getLogger("__main__")


class ForecastingRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_mae = 0
        epoch_mse = 0
        epoch_loss = 0

        for batch in self.dataloader:
            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            if self.mixup is not None:
                X, targets_a, targets_b, lam = mixup_data(
                    X, targets, self.mixup, use_cuda=True
                )

            predictions = self.model(X, padding_masks)

            if self.mixup is not None:
                loss = mixup_criterion(
                    self.criterion, predictions, targets_a, targets_b, lam
                )
            else:
                loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # loss, can be mse or hierarchical mse
            epoch_loss += loss.item()

            # hierarchical linear output
            if len(predictions) == 2:
                predictions = torch.stack(predictions[1]).sum(0)
                predictions = predictions.permute(0, 2, 1)

            # mse and mae
            epoch_mse += l2_loss(predictions, targets).item()
            epoch_mae += l1_loss(predictions, targets).item()

        # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)
        self.epoch_metrics["mae"] = epoch_mae / len(self.dataloader)
        self.epoch_metrics["mse"] = epoch_mse / len(self.dataloader)

        if self.scheduler:
            self.scheduler.step()

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model = self.model.eval()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_loss = 0
        epoch_mse = 0
        epoch_mae = 0

        for batch in self.dataloader:
            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            predictions = self.model(X, padding_masks)
            loss = self.criterion(predictions, targets)

            # hierarchical linear output
            if len(predictions) == 2:
                predictions = torch.stack(predictions[1]).sum(0)
                predictions = predictions.permute(0, 2, 1)

            epoch_loss += loss.item()

            # mse and mae
            epoch_mse += l2_loss(predictions, targets).item()
            epoch_mae += l1_loss(predictions, targets).item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)
        self.epoch_metrics["mae"] = epoch_mae / len(self.dataloader)
        self.epoch_metrics["mse"] = epoch_mse / len(self.dataloader)

        return self.epoch_metrics
