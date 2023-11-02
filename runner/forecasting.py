# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import torch
from runner.base import BaseRunner
from data.dataset import create_patch

logger = logging.getLogger("__main__")


class ForecastingRunner(BaseRunner):
    def __init__(
        self,
        model,
        dataloader,
        device,
        criterion,
        print_interval,
        console,
        use_time_features,
        layer_wise_prediction,
        hierarchical_loss,
        patch_len,
        stride,
        differencing,
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

        self.use_time_features = use_time_features
        self.layer_wise_prediction = layer_wise_prediction
        self.hierarchical_loss = hierarchical_loss

        self.patch_len = patch_len
        self.stride = stride

        self.differencing = differencing

    def train_epoch(self, epoch_num=None):
        self.model.train()
        l1_loss = torch.nn.L1Loss(reduction="mean")
        l2_loss = torch.nn.MSELoss(reduction="mean")

        epoch_loss = 0
        epoch_mae = 0
        epoch_mse = 0
        num_samples = 0

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

            # bs, seq_len, n_features = X.shape
            # X = X.transpose(1, 2).reshape(bs * n_features, seq_len)
            # targets = targets.transpose(1, 2).reshape(bs * n_features, -1)

            # targets = targets.to(torch.float64)
            # X = X.to(torch.float64)

            # with lag = 1
            # if self.differencing:
            #     repeats = 2
            #     X_orig = X
            #     targets_orig = tyargets

            #     pred_start = []
            #     targets = torch.cat([X_orig[:, -repeats:, :], targets], dim=1)

            #     for i in range(repeats):
            #         X = torch.diff(X, n=1, dim=1)
            #         pred_start.append(targets[:, :1, :])
            #         targets = torch.diff(targets, n=1, dim=1)

            # min = -1
            # max = 1
            # max_min_diff = X.max(dim=1, keepdim=True)[0] - X.min(dim=1, keepdim=True)[0]
            # input_min = X.min(dim=1, keepdim=True)[0]

            # # normalization

            # X = (X - input_min) / max_min_diff
            # X = X * (max - min) + min

            # targets = (targets - input_min) / max_min_diff
            # targets = targets * (max - min) + min

            X = create_patch(X, self.patch_len, self.stride)

            if X_time is not None and y_time is not None:
                X_time = create_patch(X_time, self.patch_len, self.stride)
                y_time = create_patch(y_time, self.patch_len, self.stride)

            # if self.use_time_features:
            predictions = self.model(
                X,
                X_time=X_time,
                y_time=y_time,
                padding_mask=padding_masks,
            )

            # de-normalizations

            # if self.differencing:
            #     targets = targets_orig

            #     # denormalization prediction
            #     predictions = (predictions - min) / (max - min)
            #     predictions = predictions * max_min_diff + input_min

            #     pred_start = pred_start[::-1]

            #     for i in range(repeats):
            #         predictions = torch.cat([pred_start[i], predictions], dim=1)
            #         predictions = torch.cumsum(predictions, dim=1, dtype=torch.float64)

            #     predictions = predictions[:, -targets.shape[1] :, :]

            if self.layer_wise_prediction:
                if self.hierarchical_loss:
                    with torch.no_grad():
                        targets_revin = self.model.revin_layer(targets, mode="norm_y")
                    targets_revin = targets

                    loss = self.criterion(predictions[1], targets_revin)
                else:
                    loss = self.criterion(predictions[0], targets)
            else:
                loss = self.criterion(predictions, targets)

            # if self.mixup is not None:
            #     loss = mixup_criterion(
            #         self.criterion, predictions, targets_a, targets_b, lam
            #     )
            # else:
            #     loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # hierarchical output
            if self.layer_wise_prediction:
                predictions = predictions[0]
            elif self.differencing:
                targets = targets_orig

                # denormalization prediction
                # predictions = (predictions - min) / (max - min)
                # predictions = predictions * max_min_diff + input_min

                pred_start = pred_start[::-1]

                for i in range(repeats):
                    predictions = torch.cat([pred_start[i], predictions], dim=1)
                    predictions = torch.cumsum(predictions, dim=1, dtype=torch.float64)

                predictions = predictions[:, -targets.shape[1] :, :]

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

            # targets = targets.to(torch.float64)
            # X = X.to(torch.float64)

            # with lag = 1
            if self.differencing:
                repeats = 2
                X_orig = X
                targets_orig = targets

                pred_start = []
                targets = torch.cat([X_orig[:, -repeats:, :], targets], dim=1)

                for i in range(repeats):
                    X = torch.diff(X, n=1, dim=1)
                    pred_start.append(targets[:, :1, :])
                    targets = torch.diff(targets, n=1, dim=1)

            # min = -1
            # max = 1
            # max_min_diff = X.max(dim=1, keepdim=True)[0] - X.min(dim=1, keepdim=True)[0]
            # input_min = X.min(dim=1, keepdim=True)[0]

            # # normalization

            # X = (X - input_min) / max_min_diff
            # X = X * (max - min) + min

            # targets = (targets - input_min) / max_min_diff
            # targets = targets * (max - min) + min

            X = create_patch(X, self.patch_len, self.stride)

            if X_time is not None and y_time is not None:
                X_time = create_patch(X_time, self.patch_len, self.stride)
                y_time = create_patch(y_time, self.patch_len, self.stride)

            # if self.use_time_features:
            predictions = self.model(
                X,
                X_time=X_time,
                y_time=y_time,
                padding_mask=padding_masks,
            )

            if self.layer_wise_prediction:
                if self.hierarchical_loss:
                    with torch.no_grad():
                        targets_revin = self.model.revin_layer(targets, mode="norm_y")
                    targets_revin = targets

                    loss = self.criterion(predictions[1], targets_revin)
                else:
                    loss = self.criterion(predictions[0], targets)
            else:
                loss = self.criterion(predictions, targets)

            # hierarchical output
            if self.layer_wise_prediction:
                predictions = predictions[0]
            elif self.differencing:
                targets = targets_orig

                # denormalization prediction
                # predictions = (predictions - min) / (max - min)
                # predictions = predictions * max_min_diff + input_min

                pred_start = pred_start[::-1]

                for i in range(repeats):
                    predictions = torch.cat([pred_start[i], predictions], dim=1)
                    predictions = torch.cumsum(predictions, dim=1, dtype=torch.float64)

                predictions = predictions[:, -targets.shape[1] :, :]

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
