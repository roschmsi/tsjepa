# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import torch
from runner.base import BaseRunner
from models.patch_tst_decomposition.decomposition import series_decomp

logger = logging.getLogger("__main__")


class DecompositionRunner(BaseRunner):
    def __init__(
        self,
        model,
        dataloader,
        device,
        criterion,
        print_interval,
        console,
        optimizer=None,
        scheduler=None,
        trend_seasonal_residual=False,
        moving_avg=None,
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

        self.trend_seasonal_residual = trend_seasonal_residual
        if not self.trend_seasonal_residual:
            self.series_decomp = series_decomp(kernel_size=int(moving_avg))

    def train_epoch(self, epoch_num=None):
        self.model.train()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_loss = 0
        epoch_mae = 0
        epoch_mse = 0
        num_samples = 0

        for batch in self.dataloader:
            if self.trend_seasonal_residual:
                trend, seasonal, residual, targets = batch
            else:
                X, targets, padding_mask = batch
                trend, residual = self.series_decomp(X)
                seasonal = None

            trend = trend.to(self.device)
            if seasonal is not None:
                seasonal = seasonal.to(self.device)
            residual = residual.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(
                trend,
                seasonal,
                residual,
                padding_mask=None,
            )

            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # mse and mae
            epoch_mse += l2_loss(predictions, targets).item() * predictions.shape[0]
            epoch_mae += l1_loss(predictions, targets).item() * predictions.shape[0]
            epoch_loss += loss.item() * predictions.shape[0]
            num_samples += predictions.shape[0]

        # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / num_samples
        self.epoch_metrics["mae"] = epoch_mae / num_samples
        self.epoch_metrics["mse"] = epoch_mse / num_samples

        if self.scheduler is not None:
            self.scheduler.step()

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model.eval()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_loss = 0
        epoch_mae = 0
        epoch_mse = 0
        num_samples = 0

        for batch in self.dataloader:
            if self.trend_seasonal_residual:
                trend, seasonal, residual, targets = batch
            else:
                X, targets, padding_mask = batch
                trend, residual = self.series_decomp(X)
                seasonal = None

            trend = trend.to(self.device)
            if seasonal is not None:
                seasonal = seasonal.to(self.device)
            residual = residual.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(
                trend,
                seasonal,
                residual,
                padding_mask=None,
            )

            loss = self.criterion(predictions, targets)

            # mse and mae
            epoch_mse += l2_loss(predictions, targets).item() * predictions.shape[0]
            epoch_mae += l1_loss(predictions, targets).item() * predictions.shape[0]
            epoch_loss += loss.item() * predictions.shape[0]
            num_samples += predictions.shape[0]

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / num_samples
        self.epoch_metrics["mae"] = epoch_mae / num_samples
        self.epoch_metrics["mse"] = epoch_mse / num_samples

        return self.epoch_metrics
