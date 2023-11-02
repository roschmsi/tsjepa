# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import torch
from runner.base import BaseRunner

logger = logging.getLogger("__main__")


class MultiresRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):
        self.model.train()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_mae = 0
        epoch_mse = 0
        epoch_loss = 0

        for batch in self.dataloader:
            if self.use_time_features:
                X, targets, padding_masks, X_time, y_time = batch
                X_time = X_time.to(self.device)
                y_time = y_time.to(self.device)
            else:
                X, targets, padding_masks = batch
                X_time = None
                y_time = None

            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            enc_target, enc_pred, dec_target, dec_pred = self.model(
                X, targets, padding_masks, X_time=X_time, y_time=y_time
            )

            loss = 0

            for i in range(len(enc_pred)):
                loss += self.criterion(enc_pred[i], enc_target[i])
                loss += self.criterion(dec_pred[i], dec_target[i])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # loss, can be mse or hierarchical mse
            epoch_loss += loss.item()

            predictions = torch.stack(dec_pred, dim=0).sum(dim=0)

            # mse and mae
            epoch_mse += l2_loss(predictions, targets).item()
            epoch_mae += l1_loss(predictions, targets).item()

        # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)
        self.epoch_metrics["mae"] = epoch_mae / len(self.dataloader)
        self.epoch_metrics["mse"] = epoch_mse / len(self.dataloader)

        if self.scheduler is not None:
            self.scheduler.step()

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model.eval()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_loss = 0
        epoch_mse = 0
        epoch_mae = 0

        for batch in self.dataloader:
            if self.use_time_features:
                X, targets, padding_masks, X_time, y_time = batch
                X_time = X_time.to(self.device)
                y_time = y_time.to(self.device)
            else:
                X, targets, padding_masks = batch
                X_time = None
                y_time = None

            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            enc_target, enc_pred, dec_target, dec_pred = self.model(
                X, targets, padding_masks, X_time=X_time, y_time=y_time
            )

            loss = 0

            for i in range(len(enc_pred)):
                loss += self.criterion(enc_pred[i], enc_target[i])
                loss += self.criterion(dec_pred[i], dec_target[i])

            epoch_loss += loss.item()

            # hierarchical output
            predictions = torch.stack(dec_pred, dim=0).sum(dim=0)

            # mse and mae
            epoch_mse += l2_loss(predictions, targets).item()
            epoch_mae += l1_loss(predictions, targets).item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)
        self.epoch_metrics["mae"] = epoch_mae / len(self.dataloader)
        self.epoch_metrics["mse"] = epoch_mse / len(self.dataloader)

        return self.epoch_metrics
