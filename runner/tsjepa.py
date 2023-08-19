from collections import OrderedDict
import logging
import numpy as np
import torch
from evaluation.evaluate_12ECG_score import compute_auc
from models.ts_jepa.logging import AverageMeter
from models.ts_jepa.tensors import apply_masks
from runner.base import BaseRunner
import torch.nn.functional as F
from models.ts_jepa.vic_reg import vicreg_fn

logger = logging.getLogger("__main__")


def forward_target(target_encoder, X, masks_pred):
    with torch.no_grad():
        h = target_encoder(X)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)
        return h


def forward_context(encoder, predictor, X, masks_enc, masks_pred):
    z_enc = encoder(X, masks_enc)
    z_pred = predictor(z_enc, masks_enc, masks_pred)
    return z_enc, z_pred


def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)
    # loss = AllReduce.apply(loss)
    return loss


def load_time_series(X, masks_enc, masks_pred, device):
    # -- unsupervised imgs
    imgs = X.to(device, non_blocking=True).float()
    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
    return (imgs, masks_1, masks_2)


class JEPARunner(BaseRunner):
    def __init__(
        self,
        encoder,
        predictor,
        target_encoder,
        revin,
        dataloader,
        device,
        optimizer=None,
        momentum_scheduler=None,
        mixup=0,
        print_interval=10,
        console=True,
        multilabel=False,
        scheduler=None,
        vic_reg=False,
        pred_weight=1.0,
        std_weight=1.0,
        cov_weight=1.0,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.revin = revin
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.momentum_scheduler = momentum_scheduler
        self.mixup = mixup
        self.print_interval = print_interval
        self.console = console
        self.multilabel = multilabel
        self.scheduler = scheduler
        self.vic_reg = vic_reg

        self.rec_weight = pred_weight
        self.std_weight = std_weight
        self.cov_weight = cov_weight

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.encoder = self.encoder.train()
        self.predictor = self.predictor.train()
        self.target_encoder = self.target_encoder.train()

        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_std_meter = AverageMeter()
        loss_cov_meter = AverageMeter()

        var_enc_meter = AverageMeter()
        var_pred_meter = AverageMeter()

        for batch in self.dataloader:
            X, _, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_time_series(
                X, masks_enc, masks_pred, self.device
            )

            # X: (bs x seq_len x n_vars)
            if self.revin is not None:
                X = self.revin(X, "norm")

            h = forward_target(
                target_encoder=self.target_encoder, X=X, masks_pred=masks_pred
            )
            z_enc, z_pred = forward_context(
                encoder=self.encoder,
                predictor=self.predictor,
                X=X,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
            )
            loss = loss_fn(z=z_pred, h=h)

            pred_loss = loss
            std_loss, cov_loss, cov_enc, cov_pred = vicreg_fn(
                z_enc=z_enc, z_pred=z_pred
            )
            # std_loss, cov_loss = pred_vicreg_fn(z_pred=z_pred)
            if self.vic_reg:
                loss = (
                    self.rec_weight * loss
                    + self.std_weight * std_loss
                    + self.cov_weight * cov_loss
                )

            #  Step 2. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Step 3. momentum update of target encoder
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(
                    self.encoder.parameters(), self.target_encoder.parameters()
                ):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            # track variance of prediction
            z_enc = z_enc.reshape(-1, z_enc.shape[-1])
            var_enc = torch.sqrt(z_enc.var(dim=0) + 1e-6).mean()

            z_pred = z_pred.reshape(-1, z_pred.shape[-1])
            var_pred = torch.sqrt(z_pred.var(dim=0) + 1e-6).mean()

            loss_meter.update(loss.item())
            loss_pred_meter.update(pred_loss.item())
            loss_std_meter.update(std_loss.item())
            loss_cov_meter.update(cov_loss.item())

            var_enc_meter.update(var_enc.item())
            var_pred_meter.update(var_pred.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        if self.vic_reg:
            self.epoch_metrics["loss pred"] = loss_pred_meter.avg
            self.epoch_metrics["loss std"] = loss_std_meter.avg
            self.epoch_metrics["loss cov"] = loss_cov_meter.avg

        self.epoch_metrics["encoder variance"] = var_enc_meter.avg
        self.epoch_metrics["pred variance"] = var_pred_meter.avg

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.encoder = self.encoder.eval()
        self.predictor = self.predictor.eval()
        self.target_encoder = self.target_encoder.eval()

        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_std_meter = AverageMeter()
        loss_cov_meter = AverageMeter()

        var_enc_meter = AverageMeter()
        var_pred_meter = AverageMeter()

        for batch in self.dataloader:
            X, _, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_time_series(
                X, masks_enc, masks_pred, self.device
            )

            # X: (bs x n_vars x seq_len)
            if self.revin is not None:
                X = self.revin(X, "norm")

            h = forward_target(
                target_encoder=self.target_encoder, X=X, masks_pred=masks_pred
            )
            z_enc, z_pred = forward_context(
                encoder=self.encoder,
                predictor=self.predictor,
                X=X,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
            )
            loss = loss_fn(z=z_pred, h=h)

            pred_loss = loss
            std_loss, cov_loss, cov_enc, cov_pred = vicreg_fn(
                z_enc=z_enc, z_pred=z_pred
            )
            # std_loss, cov_loss = pred_vicreg_fn(z_pred=z_pred)
            if self.vic_reg:
                loss = (
                    self.rec_weight * loss
                    + self.std_weight * std_loss
                    + self.cov_weight * cov_loss
                )

            # track variance of prediction
            z_enc = z_enc.reshape(-1, z_enc.shape[-1])
            var_enc = torch.sqrt(z_enc.var(dim=0) + 1e-6).mean()

            z_pred = z_pred.reshape(-1, z_pred.shape[-1])
            var_pred = torch.sqrt(z_pred.var(dim=0) + 1e-6).mean()

            loss_meter.update(loss.item())
            loss_pred_meter.update(pred_loss.item())
            loss_std_meter.update(std_loss.item())
            loss_cov_meter.update(cov_loss.item())

            var_enc_meter.update(var_enc.item())
            var_pred_meter.update(var_pred.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        if self.vic_reg:
            self.epoch_metrics["loss pred"] = loss_pred_meter.avg
            self.epoch_metrics["loss std"] = loss_std_meter.avg
            self.epoch_metrics["loss cov"] = loss_cov_meter.avg

        self.epoch_metrics["encoder variance"] = var_enc_meter.avg
        self.epoch_metrics["pred variance"] = var_pred_meter.avg

        return self.epoch_metrics


class JEPAClassifier(BaseRunner):
    def __init__(
        self,
        model,
        revin,
        dataloader,
        device,
        multilabel,
        criterion,
        optimizer=None,
    ):
        self.model = model
        self.revin = revin
        self.dataloader = dataloader
        self.device = device
        self.multilabel = multilabel
        self.criterion = criterion
        self.optimizer = optimizer

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        lbls = []
        probs = []

        for batch in self.dataloader:
            X, y, masks_enc, masks_pred = batch
            X = X.to(self.device).float()
            y = y.to(self.device)
            # y = torch.where(y == 1)[1]

            # X: (bs x n_vars x seq_len)
            if self.revin is not None:
                X = self.revin(X, "norm")

            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)

            #  Step 2. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())

            if self.multilabel:
                prob = y_pred.sigmoid().data.cpu().numpy()
                probs.append(prob)
                lbls.append(y.data.cpu().numpy())
            else:
                acc = (y_pred.argmax(dim=1) == y).sum() / y.shape[0]
                acc_meter.update(acc.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        if self.multilabel:
            lbls = np.concatenate(lbls)
            probs = np.concatenate(probs)
            auroc, _ = compute_auc(lbls, probs)
            self.epoch_metrics["auroc"] = auroc
        else:
            self.epoch_metrics["acc"] = acc_meter.avg

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model = self.model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        lbls = []
        probs = []

        for batch in self.dataloader:
            X, y, masks_enc, masks_pred = batch
            X = X.to(self.device).float()
            y = y.to(self.device)
            # y = torch.where(y == 1)[1]

            # X: (bs x n_vars x seq_len)
            if self.revin is not None:
                X = self.revin(X, "norm")

            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss_meter.update(loss.item())

            if self.multilabel:
                prob = y_pred.sigmoid().data.cpu().numpy()
                probs.append(prob)
                lbls.append(y.data.cpu().numpy())
            else:
                acc = (y_pred.argmax(dim=1) == y).sum() / y.shape[0]
                acc_meter.update(acc.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        if self.multilabel:
            lbls = np.concatenate(lbls)
            probs = np.concatenate(probs)
            auroc, _ = compute_auc(lbls, probs)
            self.epoch_metrics["auroc"] = auroc
        else:
            self.epoch_metrics["acc"] = acc_meter.avg

        return self.epoch_metrics


class JEPAForecaster(BaseRunner):
    def __init__(
        self,
        model,
        revin,
        dataloader,
        device,
        criterion,
        optimizer=None,
    ):
        self.model = model
        self.revin = revin
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.l1_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_loss = torch.nn.MSELoss(reduction="mean")

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        for batch in self.dataloader:
            X, y, masks_enc, masks_pred = batch
            X = X.to(self.device).float()
            y = y.to(self.device)

            # X: (bs x n_vars x seq_len)
            if self.revin is not None:
                X = self.revin(X, "norm")

            y_pred = self.model(X)

            if self.revin is not None:
                y_pred = self.revin(y_pred, "denorm")

            loss = self.criterion(y_pred, y)

            #  Step 2. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())
            mse = self.l2_loss(y_pred, y)
            mse_meter.update(mse.item())
            mae = self.l1_loss(y_pred, y)
            mae_meter.update(mae.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["mse"] = mse_meter.avg
        self.epoch_metrics["mae"] = mae_meter.avg

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model = self.model.eval()

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        for batch in self.dataloader:
            X, y, masks_enc, masks_pred = batch
            X = X.to(self.device).float()
            y = y.to(self.device)

            # X: (bs x n_vars x seq_len)
            if self.revin is not None:
                X = self.revin(X, "norm")

            y_pred = self.model(X)

            if self.revin is not None:
                y_pred = self.revin(y_pred, "denorm")

            loss = self.criterion(y_pred, y)

            loss_meter.update(loss.item())
            mse = self.l2_loss(y_pred, y)
            mse_meter.update(mse.item())
            mae = self.l1_loss(y_pred, y)
            mae_meter.update(mae.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["mse"] = mse_meter.avg
        self.epoch_metrics["mae"] = mae_meter.avg

        return self.epoch_metrics
