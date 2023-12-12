from collections import OrderedDict
import logging
import numpy as np
import torch
from evaluation.evaluate_12ECG_score import compute_auc
from models.ts_jepa.logging import AverageMeter
from models.ts_jepa.tensors import apply_masks
from runner.base import BaseRunner
import torch.nn.functional as F
from models.ts_jepa.vic_reg import (
    vicreg_fn,
    vibcreg_fn,
    enc_vicreg_fn,
    vicreg,
)
from models.ts_jepa.distributed import AllReduce
import matplotlib.pyplot as plt

logger = logging.getLogger("__main__")


def plot_cov_matrix(cov_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cov_matrix.detach(), vmin=-1, vmax=1)
    fig.colorbar(cax)
    return fig


def forward_target(
    target_encoder,
    X,
    masks_pred,
    return_layerwise_representations=False,
    normalize_targets=False,
):
    with torch.no_grad():
        if return_layerwise_representations:
            h, layerwise_rep = target_encoder(X, return_layerwise_representations=True)
        else:
            h = target_encoder(X)
        if normalize_targets:
            h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)
        return h


def forward_context(encoder, predictor, X, masks_enc, masks_pred):
    z_enc = encoder(X, masks_enc)
    z_pred = predictor(z_enc, masks_enc, masks_pred)
    return z_enc, z_pred


def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)
    # loss = F.mse_loss(z, h)
    # loss = AllReduce.apply(loss)
    return loss


def instance_normalization(h):
    h = h.transpose(1, 2)
    h = F.instance_norm(h)
    h = h.transpose(1, 2)
    return h


def load_time_series(X, masks_enc, masks_pred, device):
    X = X.to(device, non_blocking=True).float()
    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
    return (X, masks_1, masks_2)


class JEPARunner(BaseRunner):
    def __init__(
        self,
        encoder,
        predictor,
        revin,
        dataloader,
        device,
        optimizer=None,
        scheduler=None,
        ema=False,
        momentum_scheduler=None,
        target_encoder=None,
        vic_reg=False,
        pred_weight=1.0,
        std_weight=0.0,
        cov_weight=0.0,
    ):
        self.encoder = encoder
        self.predictor = predictor

        self.revin = revin
        self.dataloader = dataloader
        self.device = device

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.ema = ema
        self.momentum_scheduler = momentum_scheduler
        self.target_encoder = target_encoder

        self.vic_reg = vic_reg
        self.pred_weight = pred_weight
        self.std_weight = std_weight
        self.cov_weight = cov_weight

        self.epoch_metrics = OrderedDict()
        self.epoch_imgs = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.encoder.train()
        self.predictor.train()

        if self.ema:
            self.target_encoder.train()

        # track loss
        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_std_meter = AverageMeter()
        loss_cov_meter = AverageMeter()

        # track standard deviation
        enc_std_meter = AverageMeter()
        pred_std_meter = AverageMeter()
        target_std_meter = AverageMeter()

        # track covariance
        encoder_cov_matrix = torch.zeros(self.encoder.embed_dim, self.encoder.embed_dim)
        predictor_cov_matrix = torch.zeros(
            self.predictor.embed_dim, self.predictor.embed_dim
        )
        target_cov_matrix = torch.zeros(self.encoder.embed_dim, self.encoder.embed_dim)

        for batch in self.dataloader:
            X, _, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_time_series(
                X, masks_enc, masks_pred, self.device
            )

            # X: (bs x seq_len x n_vars)
            if self.revin is not None:
                X = self.revin(X, "norm")

            z_enc, z_pred = forward_context(
                encoder=self.encoder,
                predictor=self.predictor,
                X=X,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
            )

            if self.ema:
                z_target = self.target_encoder(X)
                h_unnormalized = z_target
                # TODO layer norm or instance norm
                # h = F.layer_norm(h, (h.size(-1),))
                z_target = instance_normalization(z_target)
                z_target = apply_masks(z_target, masks_pred)
            else:
                # important to forward full X for contextualized target representations
                z_target = self.encoder(X)
                h_unnormalized = z_target
                z_target = instance_normalization(z_target)
                z_target = apply_masks(z_target, masks_pred)

            loss = loss_fn(z=z_pred, h=z_target)

            pred_loss = loss

            # only compute vicreg for tracking
            (
                enc_std_loss,
                enc_cov_loss,
                enc_std,
                enc_cov,
            ) = vicreg(z_enc)
            (
                pred_std_loss,
                pred_cov_loss,
                pred_std,
                pred_cov,
            ) = vicreg(z_pred)
            (
                target_std_loss,
                target_cov_loss,
                target_std,
                target_cov,
            ) = vicreg(h_unnormalized)

            # if self.vic_reg:
            if self.target_encoder is None:
                # TODO what if not 50 % masked?
                std_loss = enc_std_loss + pred_std_loss + target_std_loss
                cov_loss = enc_cov_loss + pred_cov_loss + target_cov_loss
            else:
                std_loss = enc_std_loss + pred_std_loss
                cov_loss = enc_cov_loss + pred_cov_loss
            # elif self.vibc_reg:
            #     std_loss, cov_loss, cov_enc, cov_pred = vibcreg_fn(
            #         z_enc=z_enc, z_pred=z_pred
            #     )
            # elif self.vic_reg_enc:
            #     std_loss, cov_loss = enc_vicreg_fn(z_enc=z_enc)

            # actually apply vicreg
            if self.vic_reg or self.vibc_reg or self.vic_reg_enc:
                loss = (
                    self.pred_weight * loss
                    + self.std_weight * std_loss
                    + self.cov_weight * cov_loss
                )

            # loss = AllReduce.apply(loss)

            #  Step 2. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step 3. momentum update of target encoder
            if self.momentum_scheduler is not None:
                with torch.no_grad():
                    m = next(self.momentum_scheduler)
                    for param_q, param_k in zip(
                        self.encoder.parameters(), self.target_encoder.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            # track variance of prediction
            loss_meter.update(loss.item(), n=X.shape[0])
            loss_pred_meter.update(pred_loss.item(), n=X.shape[0])
            loss_std_meter.update(std_loss.item(), n=X.shape[0])
            loss_cov_meter.update(cov_loss.item(), n=X.shape[0])

            enc_std_meter.update(enc_std.item(), n=X.shape[0])
            pred_std_meter.update(pred_std.item(), n=X.shape[0])
            target_std_meter.update(target_std.item(), n=X.shape[0])

            encoder_cov_matrix += enc_cov.detach().cpu()
            predictor_cov_matrix += pred_cov.detach().cpu()
            target_cov_matrix += target_cov.detach().cpu()

        # scheduler for learning rate change every epoch
        if self.scheduler is not None:
            self.scheduler.step()

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        if self.vic_reg:  #  or self.vibc_reg or self.vic_reg_enc:
            self.epoch_metrics["loss prediction"] = loss_pred_meter.avg
            self.epoch_metrics["loss standard deviation"] = loss_std_meter.avg
            self.epoch_metrics["loss covariance"] = loss_cov_meter.avg

            encoder_cov_matrix = encoder_cov_matrix / len(self.dataloader)
            predictor_cov_matrix = predictor_cov_matrix / len(self.dataloader)

            self.epoch_imgs["covariance matrix encoder"] = plot_cov_matrix(
                encoder_cov_matrix
            )
            self.epoch_imgs["covariance matrix decoder"] = plot_cov_matrix(
                predictor_cov_matrix
            )
            self.epoch_imgs["covariance matrix target encoder "] = plot_cov_matrix(
                target_cov_matrix
            )

        self.epoch_metrics["standard deviation encoder"] = enc_std_meter.avg
        self.epoch_metrics["standard deviation decoder"] = pred_std_meter.avg
        self.epoch_metrics["standard deviation target encoder"] = target_std_meter.avg

        return self.epoch_metrics, self.epoch_imgs

    def evaluate(self, epoch_num=None):
        self.encoder = self.encoder.eval()
        self.predictor = self.predictor.eval()
        if self.target_encoder is not None:
            self.target_encoder.eval()

        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_std_meter = AverageMeter()
        loss_cov_meter = AverageMeter()
        enc_std_meter = AverageMeter()
        pred_std_meter = AverageMeter()
        target_std_meter = AverageMeter()

        encoder_cov_matrix = torch.zeros(self.encoder.embed_dim, self.encoder.embed_dim)
        predictor_cov_matrix = torch.zeros(
            self.predictor.embed_dim, self.predictor.embed_dim
        )
        target_cov_matrix = torch.zeros(self.encoder.embed_dim, self.encoder.embed_dim)

        for batch in self.dataloader:
            X, _, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_time_series(
                X, masks_enc, masks_pred, self.device
            )

            # X: (bs x n_vars x seq_len)
            if self.revin is not None:
                X = self.revin(X, masks_enc, masks_pred)

            if self.target_encoder is not None:
                h = self.target_encoder(X)
                h_unnormalized = h
                # TODO layer norm or instance norm
                # h = F.layer_norm(h, (h.size(-1),))
                h = instance_normalization(h)
                h = apply_masks(h, masks_pred)
            else:
                h = self.encoder(X)
                h_unnormalized = h
                h = instance_normalization(h)
                h = apply_masks(h, masks_pred)

            z_enc, z_pred = forward_context(
                encoder=self.encoder,
                predictor=self.predictor,
                X=X,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
            )
            loss = loss_fn(z=z_pred, h=h)

            pred_loss = loss

            # only compute vicreg for tracking
            (
                enc_std_loss,
                enc_cov_loss,
                enc_std,
                enc_cov,
            ) = vicreg(z_enc)
            (
                pred_std_loss,
                pred_cov_loss,
                pred_std,
                pred_cov,
            ) = vicreg(z_pred)
            (
                target_std_loss,
                target_cov_loss,
                target_std,
                target_cov,
            ) = vicreg(h_unnormalized)

            num_vectors = z_enc.shape[0] + z_pred.shape[0] + h.shape[0]

            if self.target_encoder is None:
                # TODO what if not 50 % masked?
                std_loss = enc_std_loss + pred_std_loss + target_std_loss
                cov_loss = enc_cov_loss + pred_cov_loss + target_cov_loss
            else:
                std_loss = enc_std_loss + pred_std_loss
                cov_loss = enc_cov_loss + pred_cov_loss

            # actually apply vicreg
            if self.vic_reg:  #  or self.vibc_reg or self.vic_reg_enc:
                loss = (
                    self.pred_weight * loss
                    + self.std_weight * std_loss
                    + self.cov_weight * cov_loss
                )

            # track variance of prediction
            z_enc = z_enc.reshape(-1, z_enc.shape[-1])
            var_enc = torch.sqrt(z_enc.var(dim=0) + 1e-6).mean()

            z_pred = z_pred.reshape(-1, z_pred.shape[-1])
            var_pred = torch.sqrt(z_pred.var(dim=0) + 1e-6).mean()

            loss_meter.update(loss.item(), n=X.shape[0])
            loss_pred_meter.update(pred_loss.item(), n=X.shape[0])
            loss_std_meter.update(std_loss.item(), n=X.shape[0])
            loss_cov_meter.update(cov_loss.item(), n=X.shape[0])

            var_enc_meter.update(var_enc.item(), n=X.shape[0])
            var_pred_meter.update(var_pred.item(), n=X.shape[0])

            encoder_cov_matrix += cov_enc.detach().cpu()
            predictor_cov_matrix += cov_pred.detach().cpu()

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        if self.vic_reg:  #  or self.vibc_reg or self.vic_reg_enc:
            self.epoch_metrics["loss prediction"] = loss_pred_meter.avg
            self.epoch_metrics["loss standard deviation"] = loss_std_meter.avg
            self.epoch_metrics["loss covariance"] = loss_cov_meter.avg

            encoder_cov_matrix = encoder_cov_matrix / len(self.dataloader)
            predictor_cov_matrix = predictor_cov_matrix / len(self.dataloader)

            self.epoch_imgs["covariance matrix encoder"] = plot_cov_matrix(
                encoder_cov_matrix
            )
            self.epoch_imgs["covariance matrix decoder"] = plot_cov_matrix(
                predictor_cov_matrix
            )
            self.epoch_imgs["covariance matrix target encoder "] = plot_cov_matrix(
                target_cov_matrix
            )

        self.epoch_metrics["standard deviation encoder"] = enc_std_meter.avg
        self.epoch_metrics["standard deviation decoder"] = pred_std_meter.avg
        self.epoch_metrics["standard deviation target encoder"] = target_std_meter.avg

        return self.epoch_metrics, self.epoch_imgs


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
        scheduler=None,
    ):
        self.model = model
        self.revin = revin
        self.dataloader = dataloader
        self.device = device
        self.multilabel = multilabel
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.model.train()

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

        # scheduler for learning rate change every epoch
        if self.scheduler is not None:
            self.scheduler.step()

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
        self.model.eval()

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
        scheduler=None,
    ):
        self.model = model
        self.revin = revin
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.l1_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_loss = torch.nn.MSELoss(reduction="mean")

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.model.train()

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        for batch in self.dataloader:
            X, y, masks_enc, masks_pred = batch
            X = X.to(self.device).float()
            y = y.to(self.device)

            # X: (bs x n_vars x seq_len)
            # channel independence for the exact same setting
            bs, seq_len, n_vars = X.shape

            if self.revin is not None:
                X = X.transpose(1, 2)
                X = X.reshape(bs * n_vars, seq_len)
                X = X.unsqueeze(-1)
                X = self.revin(X, "norm")
                X = X.squeeze(-1)
                X = X.reshape(bs, n_vars, seq_len)
                X = X.transpose(1, 2)

            y_pred = self.model(X)
            bs, pred_len, n_vars = y_pred.shape

            if self.revin is not None:
                y_pred = y_pred.transpose(1, 2)
                y_pred = y_pred.reshape(bs * n_vars, pred_len)
                y_pred = y_pred.unsqueeze(-1)
                y_pred = self.revin(y_pred, "denorm")
                y_pred = y_pred.squeeze(-1)
                y_pred = y_pred.reshape(bs, n_vars, pred_len)
                y_pred = y_pred.transpose(1, 2)

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
