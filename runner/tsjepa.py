from collections import OrderedDict
import logging
import torch
from models.ts_jepa.logging import AverageMeter
from models.ts_jepa.tensors import apply_masks
from runner.base import BaseRunner
import torch.nn.functional as F

logger = logging.getLogger("__main__")


def forward_target(target_encoder, X, masks_pred):
    with torch.no_grad():
        h = target_encoder(X)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)
        return h


def forward_context(encoder, predictor, X, masks_enc, masks_pred):
    z = encoder(X, masks_enc)
    z = predictor(z, masks_enc, masks_pred)
    return z


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
        dataloader,
        device,
        optimizer=None,
        momentum_scheduler=None,
        mixup=0,
        print_interval=10,
        console=True,
        multilabel=False,
        scheduler=None,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.momentum_scheduler = momentum_scheduler
        self.mixup = mixup
        self.print_interval = print_interval
        self.console = console
        self.multilabel = multilabel
        self.scheduler = scheduler

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        self.encoder = self.encoder.train()
        self.predictor = self.predictor.train()
        self.target_encoder = self.target_encoder.train()

        loss_meter = AverageMeter()
        var_meter = AverageMeter()

        for batch in self.dataloader:
            X, _, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_time_series(
                X, masks_enc, masks_pred, self.device
            )

            h = forward_target(
                target_encoder=self.target_encoder, X=X, masks_pred=masks_pred
            )
            z = forward_context(
                encoder=self.encoder,
                predictor=self.predictor,
                X=X,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
            )
            loss = loss_fn(z=z, h=h)

            #  Step 2. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step 3. momentum update of target encoder
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(
                    self.encoder.parameters(), self.target_encoder.parameters()
                ):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            # track variance of prediction
            pred = z.reshape(-1, z.shape[-1])
            var = torch.sqrt(pred.var(dim=0) + 1e-6).mean()
            var_meter.update(var.item())

            loss_meter.update(loss)
            var_meter.update(var.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["var"] = var_meter.avg

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.encoder = self.encoder.eval()
        self.predictor = self.predictor.eval()
        self.target_encoder = self.target_encoder.eval()

        loss_meter = AverageMeter()
        var_meter = AverageMeter()

        for batch in self.dataloader:
            X, _, masks_enc, masks_pred = batch
            X, masks_enc, masks_pred = load_time_series(
                X, masks_enc, masks_pred, self.device
            )

            h = forward_target(
                target_encoder=self.target_encoder, X=X, masks_pred=masks_pred
            )
            z = forward_context(
                encoder=self.encoder,
                predictor=self.predictor,
                X=X,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
            )
            loss = loss_fn(z=z, h=h)

            # track variance of prediction
            pred = z.reshape(-1, z.shape[-1])
            var = torch.sqrt(pred.var(dim=0) + 1e-6).mean()
            var_meter.update(var.item())

            loss_meter.update(loss)
            var_meter.update(var.item())

        # average loss per sample for whole epoch
        self.epoch_metrics["loss"] = loss_meter.avg
        self.epoch_metrics["var"] = var_meter.avg

        return self.epoch_metrics
