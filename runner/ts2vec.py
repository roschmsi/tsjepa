from collections import OrderedDict
import logging
import torch
from models.ts_jepa.logging import AverageMeter
from runner.base import BaseRunner
from models.ts_jepa.vic_reg import vicreg_single_batch
import matplotlib.pyplot as plt
import torch.nn as nn
from data.dataset import create_patch, random_patch_masking, block_patch_masking

logger = logging.getLogger("__main__")


def plot_cov_matrix(cov_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cov_matrix.detach(), vmin=-1, vmax=1)
    fig.colorbar(cax)
    return fig


# TODO integrate revin in encoder ?
class TS2VecRunner(BaseRunner):
    def __init__(
        self,
        model,
        revin,
        dataloader,
        device,
        patch_len,
        stride,
        masking,
        masking_ratio,
        debug,
        optimizer=None,
        scheduler=None,
    ):
        self.model = model

        self.revin = revin
        self.dataloader = dataloader
        self.device = device

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.patch_len = patch_len
        self.stride = stride

        self.masking = masking
        self.masking_ratio = masking_ratio
        self.debug = debug

        self.epoch_metrics = OrderedDict()
        self.epoch_imgs = OrderedDict()

        self.criterion = nn.SmoothL1Loss(reduction="mean")  # smooth l1
        self.criterion.to(self.device)

    def train_epoch(self, epoch_num=None):
        self.model.train()

        # track loss
        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_std_meter = AverageMeter()
        loss_cov_meter = AverageMeter()

        # track standard deviation
        pred_std_meter = AverageMeter()
        target_std_meter = AverageMeter()

        # track covariance
        pred_cov_meter = AverageMeter()
        target_cov_meter = AverageMeter()

        # track covariance
        pred_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)
        target_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)

        for batch in self.dataloader:
            X, y = batch

            # TODO revin for X and y separately
            # X: (bs x seq_len x n_vars)
            if self.revin is not None:
                X = self.revin(X, "norm")

            # create patch
            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            # random patch masking
            if self.masking == "random":
                X_masked, X_kept, mask, ids_keep, ids_restore = random_patch_masking(
                    X,
                    self.masking_ratio,
                    debug=self.debug,
                )
            elif self.masking == "block":
                X_masked, X_kept, mask, ids_keep, ids_restore = block_patch_masking(
                    X,
                    self.masking_ratio,
                    debug=self.debug,
                )
            else:
                raise NotImplementedError

            X = X.to(self.device)
            X_masked = X_masked.to(self.device)
            X_kept = X_kept.to(self.device)
            mask = mask.to(self.device)
            ids_keep = ids_keep.to(self.device)
            ids_restore = ids_restore.to(self.device)

            # channel independence
            X = X.squeeze()
            X_masked = X_masked.squeeze()

            # squeeze because channel dim is 1
            pred, target = self.model(
                X_full=X,
                X_masked=X_masked,
            )

            mask = mask.bool()

            # pred = torch.gather(pred, dim=1, index=ids_keep.repeat(1, 1, 128))
            # target = torch.gather(target, dim=1, index=ids_keep.repeat(1, 1, 128))

            masked_pred = torch.masked_select(pred, mask)
            masked_target = torch.masked_select(target, mask)
            loss = self.criterion(masked_pred, masked_target)

            # only compute vicreg for tracking
            (
                pred_std_loss,
                pred_cov_loss,
                pred_std,
                pred_cov,
            ) = vicreg_single_batch(pred)
            (
                target_std_loss,
                target_cov_loss,
                target_std,
                target_cov,
            ) = vicreg_single_batch(target)

            # TODO we should only track that shit for the masked parts, the others arent interesting

            pred_loss = loss

            std_loss = (pred_std_loss + target_std_loss) / 2
            cov_loss = (pred_cov_loss + target_cov_loss) / 2

            #  Step 2. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # track variance of prediction
            loss_meter.update(loss.item(), n=X.shape[0])
            loss_pred_meter.update(pred_loss.item(), n=X.shape[0])
            loss_std_meter.update(std_loss.item(), n=X.shape[0])
            loss_cov_meter.update(cov_loss.item(), n=X.shape[0])

            pred_std_meter.update(pred_std.item(), n=X.shape[0])
            target_std_meter.update(target_std.item(), n=X.shape[0])

            pred_cov_meter.update(pred_cov_loss.item(), n=X.shape[0])
            target_cov_meter.update(target_cov_loss.item(), n=X.shape[0])

            pred_cov_matrix += pred_cov.detach().cpu()
            target_cov_matrix += target_cov.detach().cpu()

            # ema teacher step
            self.model.ema_step()

        # scheduler for learning rate change every epoch
        if self.scheduler is not None:
            self.scheduler.step()
            self.epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        self.epoch_metrics["loss prediction"] = loss_pred_meter.avg
        self.epoch_metrics["loss standard deviation"] = loss_std_meter.avg
        self.epoch_metrics["loss covariance"] = loss_cov_meter.avg

        self.epoch_metrics["standard deviation pred"] = pred_std_meter.avg
        self.epoch_metrics["standard deviation target"] = target_std_meter.avg

        self.epoch_metrics["covariance pred"] = pred_cov_meter.avg
        self.epoch_metrics["covariance target"] = target_cov_meter.avg

        pred_cov_matrix = pred_cov_matrix / len(self.dataloader)
        target_cov_matrix = target_cov_matrix / len(self.dataloader)

        self.epoch_imgs["covariance matrix pred"] = plot_cov_matrix(pred_cov_matrix)
        self.epoch_imgs["covariance matrix target"] = plot_cov_matrix(target_cov_matrix)

        return self.epoch_metrics, self.epoch_imgs

    def evaluate(self, epoch_num=None):
        self.model.eval()

        # track loss
        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_std_meter = AverageMeter()
        loss_cov_meter = AverageMeter()

        # track standard deviation
        pred_std_meter = AverageMeter()
        target_std_meter = AverageMeter()

        # track covariance
        pred_cov_meter = AverageMeter()
        target_cov_meter = AverageMeter()

        # track covariance
        pred_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)
        target_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)

        for batch in self.dataloader:
            X, y = batch

            # X: (bs x seq_len x n_vars)
            if self.revin is not None:
                X = self.revin(X, "norm")

            # create patch
            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            # random patch masking
            if self.masking == "random":
                X_masked, X_kept, mask, ids_keep, ids_restore = random_patch_masking(
                    X,
                    self.masking_ratio,
                    debug=self.debug,
                )
            elif self.masking == "block":
                X_masked, X_kept, mask, ids_keep, ids_restore = block_patch_masking(
                    X,
                    self.masking_ratio,
                    debug=self.debug,
                )
            else:
                raise NotImplementedError

            X = X.to(self.device)
            X_masked = X_masked.to(self.device)
            X_kept = X_kept.to(self.device)
            mask = mask.to(self.device)
            ids_keep = ids_keep.to(self.device)
            ids_restore = ids_restore.to(self.device)

            # squeeze because channel dim is 1
            pred, target = self.model(
                X_full=X.squeeze(),
                X_masked=X_masked.squeeze(),
                # mask=mask.squeeze(),
            )

            mask = mask.bool()

            # pred = torch.gather(pred, dim=1, index=ids_keep.repeat(1, 1, 128))
            # target = torch.gather(target, dim=1, index=ids_keep.repeat(1, 1, 128))

            masked_pred = torch.masked_select(pred, mask)
            masked_target = torch.masked_select(target, mask)
            loss = self.criterion(masked_pred, masked_target)

            # only compute vicreg for tracking
            (
                pred_std_loss,
                pred_cov_loss,
                pred_std,
                pred_cov,
            ) = vicreg_single_batch(pred)
            (
                target_std_loss,
                target_cov_loss,
                target_std,
                target_cov,
            ) = vicreg_single_batch(target)

            # TODO we should only track that shit for the masked parts, the others arent interesting

            pred_loss = loss
            std_loss = (pred_std_loss + target_std_loss) / 2
            cov_loss = (pred_cov_loss + target_cov_loss) / 2

            # track variance of prediction
            loss_meter.update(loss.item(), n=X.shape[0])
            loss_pred_meter.update(pred_loss.item(), n=X.shape[0])
            loss_std_meter.update(std_loss.item(), n=X.shape[0])
            loss_cov_meter.update(cov_loss.item(), n=X.shape[0])

            pred_std_meter.update(pred_std.item(), n=X.shape[0])
            target_std_meter.update(target_std.item(), n=X.shape[0])

            pred_cov_meter.update(pred_cov_loss.item(), n=X.shape[0])
            target_cov_meter.update(target_cov_loss.item(), n=X.shape[0])

            pred_cov_matrix += pred_cov.detach().cpu()
            target_cov_matrix += target_cov.detach().cpu()

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        self.epoch_metrics["loss prediction"] = loss_pred_meter.avg
        self.epoch_metrics["loss standard deviation"] = loss_std_meter.avg
        self.epoch_metrics["loss covariance"] = loss_cov_meter.avg

        self.epoch_metrics["standard deviation pred"] = pred_std_meter.avg
        self.epoch_metrics["standard deviation target"] = target_std_meter.avg

        self.epoch_metrics["covariance pred"] = pred_cov_meter.avg
        self.epoch_metrics["covariance target"] = target_cov_meter.avg

        pred_cov_matrix = pred_cov_matrix / len(self.dataloader)
        target_cov_matrix = target_cov_matrix / len(self.dataloader)

        self.epoch_imgs["covariance matrix pred"] = plot_cov_matrix(pred_cov_matrix)
        self.epoch_imgs["covariance matrix target"] = plot_cov_matrix(target_cov_matrix)

        return self.epoch_metrics, self.epoch_imgs


class TS2VecForecastingRunner(BaseRunner):
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

    def train_epoch(self, epoch_num=None):
        self.model.train()

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        for batch in self.dataloader:
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # X: (bs x seq_len x n_vars)
            if self.revin is not None:
                bs, seq_len, n_vars = X.shape
                X = X.transpose(1, 2).reshape(bs * n_vars, seq_len).unsqueeze(-1)
                X = self.revin(X, "norm")
                X = X.squeeze(-1).reshape(bs, n_vars, seq_len).transpose(1, 2)

            # create patch
            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            # channel independence, happens in model
            # X = X.squeeze()

            # channel independence
            y_pred = self.model(X)

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

            #  Step 2. Backward & step
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

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["mse"] = mse_meter.avg
        self.epoch_metrics["mae"] = mae_meter.avg

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model.eval()

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        for batch in self.dataloader:
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # X: (bs x seq_len x n_vars)
            if self.revin is not None:
                X = self.revin(X, "norm")

            # create patch
            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            y_pred = self.model(X)

            if self.revin is not None:
                y_pred = self.revin(y_pred, "denorm")

            loss = self.criterion(y_pred, y)

            loss_meter.update(loss.item(), n=X.shape[0])
            mse = self.l2_loss(y_pred, y)
            mse_meter.update(mse.item(), n=X.shape[0])
            mae = self.l1_loss(y_pred, y)
            mae_meter.update(mae.item(), n=X.shape[0])

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["mse"] = mse_meter.avg
        self.epoch_metrics["mae"] = mae_meter.avg

        return self.epoch_metrics
