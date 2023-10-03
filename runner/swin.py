# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import torch
from runner.base import BaseRunner

logger = logging.getLogger("__main__")


class SwinRunner(BaseRunner):
    def __init__(
        self,
        model,
        dataloader,
        device,
        criterion,
        print_interval,
        console,
        patch_len,
        stride,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            print_interval=print_interval,
            console=console,
        )

        self.patch_len = patch_len
        self.stride = stride

    def train_epoch(self, epoch_num=None):
        self.model.train()
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

            # patching is done by the model
            # X = create_patch(X, self.patch_len, self.stride)

            predictions = self.model(X)

            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

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
            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            # done by the model
            # X = create_patch(X, self.patch_len, self.stride)

            predictions = self.model(X)

            loss = self.criterion(predictions, targets)

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
