import logging
from collections import OrderedDict
from data.masking import block_patch_masking, random_patch_masking
from utils.plot import plot_cov_matrix

import torch

from data.dataset import create_patch
from runner.base import BaseRunner
from utils.logging import AverageMeter

logger = logging.getLogger("__main__")


class TSJepaRunner(BaseRunner):
    """
    Trainer and Evaluator for TS-JEPA
    """

    def __init__(
        self,
        model,
        revin,
        dataloader,
        patch_len,
        stride,
        masking,
        masking_ratio,
        pred_weight,
        std_weight,
        cov_weight,
        device,
        debug,
        criterion,
        embedding_space=True,
        no_ema=False,
        optimizer=None,
        scheduler=None,
        regfn=None,
    ):
        self.model = model
        self.embedding_space = embedding_space

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

        self.criterion = criterion

        self.no_ema = no_ema

        self.pred_weight = pred_weight
        self.std_weight = std_weight
        self.cov_weight = cov_weight

        self.regfn = regfn

    def train_epoch(self):
        self.model.train()

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
        enc_cov_meter = AverageMeter()
        pred_cov_meter = AverageMeter()
        target_cov_meter = AverageMeter()

        # track covariance matrix
        enc_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)
        pred_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)
        target_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)

        for batch in self.dataloader:
            X = batch

            # reversible instance normalization
            if self.revin is not None:
                X = self.revin(X, "norm")

            X = create_patch(X, patch_len=self.patch_len, stride=self.stride)

            # patch masking
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
            X_kept = X_kept.squeeze()
            mask = mask.squeeze().bool()

            X_enc, pred, target = self.model(
                X_full=X,
                X_masked=X_masked,
                X_kept=X_kept,
                ids_kept=ids_keep,
                ids_restore=ids_restore,
            )

            mask = mask.unsqueeze(-1).repeat_interleave(
                repeats=target.shape[-1], dim=-1
            )
            masked_pred = torch.masked_select(pred, mask)
            masked_target = torch.masked_select(target, mask)
            loss = self.criterion(masked_pred, masked_target)

            # only compute vicreg for tracking
            (
                enc_std_loss,
                enc_cov_loss,
                enc_std,
                enc_cov,
            ) = self.regfn(X_enc)
            (
                pred_std_loss,
                pred_cov_loss,
                pred_std,
                pred_cov,
            ) = self.regfn(pred)
            (
                target_std_loss,
                target_cov_loss,
                target_std,
                target_cov,
            ) = self.regfn(target)

            pred_loss = loss

            if self.no_ema:
                std_loss = (enc_std_loss + pred_std_loss + target_std_loss) / 3
                cov_loss = (enc_cov_loss + pred_cov_loss + target_cov_loss) / 3
            else:
                std_loss = (enc_std_loss + pred_std_loss) / 2
                cov_loss = (enc_cov_loss + pred_cov_loss) / 2

            if self.pred_weight > 0:
                loss = self.pred_weight * loss
            if self.std_weight > 0:
                loss += self.std_weight * std_loss
            if self.cov_weight > 0:
                loss += self.cov_weight * cov_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # track variance of prediction
            loss_meter.update(loss.item(), n=X.shape[0])
            loss_pred_meter.update(pred_loss.item(), n=X.shape[0])
            loss_std_meter.update(std_loss.item(), n=X.shape[0])
            loss_cov_meter.update(cov_loss.item(), n=X.shape[0])

            enc_std_meter.update(enc_std.item(), n=X.shape[0])
            pred_std_meter.update(pred_std.item(), n=X.shape[0])
            target_std_meter.update(target_std.item(), n=X.shape[0])

            enc_cov_meter.update(enc_cov_loss.item(), n=X.shape[0])
            pred_cov_meter.update(pred_cov_loss.item(), n=X.shape[0])
            target_cov_meter.update(target_cov_loss.item(), n=X.shape[0])

            enc_cov_matrix += enc_cov.detach().cpu()
            if self.embedding_space:
                pred_cov_matrix += pred_cov.detach().cpu()
                target_cov_matrix += target_cov.detach().cpu()

            # ema update
            if self.embedding_space and not self.no_ema:
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

        self.epoch_metrics["standard deviation enc"] = enc_std_meter.avg
        self.epoch_metrics["standard deviation pred"] = pred_std_meter.avg
        self.epoch_metrics["standard deviation target"] = target_std_meter.avg

        self.epoch_metrics["covariance enc"] = enc_cov_meter.avg
        self.epoch_metrics["covariance pred"] = pred_cov_meter.avg
        self.epoch_metrics["covariance target"] = target_cov_meter.avg

        enc_cov_matrix = enc_cov_matrix / len(self.dataloader)
        pred_cov_matrix = pred_cov_matrix / len(self.dataloader)
        target_cov_matrix = target_cov_matrix / len(self.dataloader)

        self.epoch_imgs["covariance matrix enc"] = plot_cov_matrix(enc_cov_matrix)
        self.epoch_imgs["covariance matrix pred"] = plot_cov_matrix(pred_cov_matrix)
        self.epoch_imgs["covariance matrix target"] = plot_cov_matrix(target_cov_matrix)

        return self.epoch_metrics, self.epoch_imgs

    def evaluate(self):
        self.model.eval()

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
        enc_cov_meter = AverageMeter()
        pred_cov_meter = AverageMeter()
        target_cov_meter = AverageMeter()

        # track covariance
        enc_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)
        pred_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)
        target_cov_matrix = torch.zeros(self.model.embed_dim, self.model.embed_dim)

        for batch in self.dataloader:
            X = batch  # no y needed

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
            X_kept = X_kept.squeeze()
            mask = mask.squeeze().bool()

            X_enc, pred, target = self.model(
                X_full=X,
                X_masked=X_masked,
                X_kept=X_kept,
                ids_kept=ids_keep,
                ids_restore=ids_restore,
            )

            mask = mask.unsqueeze(-1).repeat_interleave(
                repeats=target.shape[-1], dim=-1
            )
            masked_pred = torch.masked_select(pred, mask)
            masked_target = torch.masked_select(target, mask)
            loss = self.criterion(masked_pred, masked_target)

            # only compute vicreg for tracking
            (
                enc_std_loss,
                enc_cov_loss,
                enc_std,
                enc_cov,
            ) = self.regfn(X_enc)
            (
                pred_std_loss,
                pred_cov_loss,
                pred_std,
                pred_cov,
            ) = self.regfn(pred)
            (
                target_std_loss,
                target_cov_loss,
                target_std,
                target_cov,
            ) = self.regfn(target)

            pred_loss = loss

            if self.no_ema:
                std_loss = (enc_std_loss + pred_std_loss + target_std_loss) / 3
                cov_loss = (enc_cov_loss + pred_cov_loss + target_cov_loss) / 3
            else:
                std_loss = (enc_std_loss + pred_std_loss) / 2
                cov_loss = (enc_cov_loss + pred_cov_loss) / 2

            if self.pred_weight > 0:
                loss = self.pred_weight * loss
            if self.std_weight > 0:
                loss += self.std_weight * std_loss
            if self.cov_weight > 0:
                loss += self.cov_weight * cov_loss

            # track variance of prediction
            loss_meter.update(loss.item(), n=X.shape[0])
            loss_pred_meter.update(pred_loss.item(), n=X.shape[0])
            loss_std_meter.update(std_loss.item(), n=X.shape[0])
            loss_cov_meter.update(cov_loss.item(), n=X.shape[0])

            enc_std_meter.update(enc_std.item(), n=X.shape[0])
            pred_std_meter.update(pred_std.item(), n=X.shape[0])
            target_std_meter.update(target_std.item(), n=X.shape[0])

            enc_cov_meter.update(enc_cov_loss.item(), n=X.shape[0])
            pred_cov_meter.update(pred_cov_loss.item(), n=X.shape[0])
            target_cov_meter.update(target_cov_loss.item(), n=X.shape[0])

            enc_cov_matrix += enc_cov.detach().cpu()
            if self.embedding_space:
                pred_cov_matrix += pred_cov.detach().cpu()
                target_cov_matrix += target_cov.detach().cpu()

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg

        self.epoch_metrics["loss prediction"] = loss_pred_meter.avg
        self.epoch_metrics["loss standard deviation"] = loss_std_meter.avg
        self.epoch_metrics["loss covariance"] = loss_cov_meter.avg

        self.epoch_metrics["standard deviation enc"] = enc_std_meter.avg
        self.epoch_metrics["standard deviation pred"] = pred_std_meter.avg
        self.epoch_metrics["standard deviation target"] = target_std_meter.avg

        self.epoch_metrics["covariance enc"] = enc_cov_meter.avg
        self.epoch_metrics["covariance pred"] = pred_cov_meter.avg
        self.epoch_metrics["covariance target"] = target_cov_meter.avg

        enc_cov_matrix = enc_cov_matrix / len(self.dataloader)
        pred_cov_matrix = pred_cov_matrix / len(self.dataloader)
        target_cov_matrix = target_cov_matrix / len(self.dataloader)

        self.epoch_imgs["covariance matrix enc"] = plot_cov_matrix(enc_cov_matrix)
        self.epoch_imgs["covariance matrix pred"] = plot_cov_matrix(pred_cov_matrix)
        self.epoch_imgs["covariance matrix target"] = plot_cov_matrix(target_cov_matrix)

        return self.epoch_metrics, self.epoch_imgs
