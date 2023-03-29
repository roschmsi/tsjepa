import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.mixup import mixup_criterion, mixup_data
from physionet_evaluation.evaluate_12ECG_score import compute_auc
from utils import Printer, save_model

logger = logging.getLogger("__main__")


def validate(
    val_evaluator,
    tensorboard_writer,
    config,
    best_metrics,
    best_loss,
    epoch,
):
    with torch.no_grad():
        aggr_metrics = val_evaluator.evaluate(epoch)

    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar(f"{k}/val", v, epoch)

    if aggr_metrics["loss"] < best_loss:
        best_loss = aggr_metrics["loss"]
        save_model(
            path=os.path.join(config["checkpoint_dir"], "model_best.pth"),
            epoch=epoch,
            model=val_evaluator.model,
        )
        best_metrics = aggr_metrics.copy()

    return best_metrics, best_loss


class BaseRunner(object):
    def __init__(
        self,
        model,
        dataloader,
        device,
        criterion,
        optimizer=None,
        print_interval=10,
        console=True,
        multilabel=False,
        scheduler=None,
        mixup=0,
        mae=False,
    ):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_interval = print_interval
        self.printer = Printer(console=console)
        self.multilabel = multilabel
        self.scheduler = scheduler

        self.epoch_metrics = OrderedDict()
        self.mixup = mixup
        self.mae = mae

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError("Please override in child class")

    def evaluate(self, epoch_num=None):
        raise NotImplementedError("Please override in child class")

    def print_callback(self, i_batch, metrics, prefix=""):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0

        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks = batch
            targets = targets.to(self.device)

            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # (batch_size, padded_length, feat_dim)
            predictions = self.model(X.to(self.device), padding_masks)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.criterion(predictions, targets, target_masks)
            batch_loss = torch.sum(loss) / len(loss)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            batch_loss.backward()
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            epoch_loss += batch_loss.item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = batch_loss / len(self.dataloader)
        return self.epoch_metrics

    def evaluate(self, epoch_num=None):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch

        for batch in self.dataloader:

            X, targets, target_masks, padding_masks = batch
            targets = targets.to(self.device)

            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # (batch_size, padded_length, feat_dim)
            predictions = self.model(X.to(self.device), padding_masks)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.criterion(predictions, targets, target_masks)
            batch_loss = torch.sum(loss) / len(loss)

            epoch_loss += batch_loss.item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        return self.epoch_metrics


class SupervisedRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        lbls = []
        probs = []

        for batch in self.dataloader:

            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            if self.mixup is not None:
                X, targets_a, targets_b, lam = mixup_data(
                    X, targets, self.mixup, use_cuda=True
                )

            predictions = self.model(X, padding_masks)

            # (batch_size,) loss for each sample in the batch
            if self.mixup is not None:
                loss = mixup_criterion(
                    self.criterion, predictions, targets_a, targets_b, lam
                )
            else:
                loss = self.criterion(predictions, targets)

            # mean loss (over samples) used for optimization
            batch_loss = torch.sum(loss) / len(loss)

            self.optimizer.zero_grad()
            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            self.optimizer.step()

            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

            total_samples += len(loss)
            epoch_loss += batch_loss.item()

        # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)

        if self.scheduler:
            self.scheduler.step()

        if self.multilabel:
            auroc, _ = compute_auc(lbls, probs)
        else:
            probs = torch.nn.functional.softmax(torch.from_numpy(probs), dim=1).numpy()
            auroc = roc_auc_score(lbls, probs, multi_class="ovo")

        self.epoch_metrics["auroc"] = auroc

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        lbls = []
        probs = []

        for batch in self.dataloader:

            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # classification: (batch_size, num_classes) of logits
            predictions = self.model(X, padding_masks)

            # (batch_size,) loss for each sample in the batch
            loss = self.criterion(predictions, targets)
            batch_loss = torch.sum(loss) / len(loss)

            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

            total_samples += len(loss)
            epoch_loss += batch_loss.item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)

        if self.multilabel:
            auroc, _ = compute_auc(lbls, probs)
        else:
            probs = torch.nn.functional.softmax(torch.from_numpy(probs), dim=1).numpy()
            auroc = roc_auc_score(lbls, probs, multi_class="ovo")

        self.epoch_metrics["auroc"] = auroc

        return self.epoch_metrics


class ForecastingRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

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

            epoch_loss += loss.item()

        # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        if self.scheduler:
            self.scheduler.step()

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):

        self.model = self.model.eval()

        epoch_loss = 0

        for batch in self.dataloader:

            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            predictions = self.model(X, padding_masks)
            loss = self.criterion(predictions, targets)

            epoch_loss += loss.item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        return self.epoch_metrics


class UnsupervisedPatchRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0

        for batch in self.dataloader:

            (
                X,
                X_kept,
                targets,
                target_masks,
                padding_masks,
                padding_masks_kept,
                ids_restore,
            ) = batch

            # 1s: mask and predict, 0s: unaffected input (ignore)
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)

            # (batch_size, padded_length, feat_dim)
            if self.mae:
                X_kept = X_kept.to(self.device)
                padding_masks_kept = padding_masks_kept.to(self.device)
                ids_restore = ids_restore.to(self.device)

                predictions = self.model(
                    X_kept,
                    padding_masks,
                    padding_masks_kept,
                    ids_restore,
                )
            else:
                X = X.to(self.device)

                predictions = self.model(
                    X,
                    padding_masks,
                )

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = self.criterion(predictions, targets, target_masks)

            print(batch_loss)

            self.optimizer.zero_grad()
            batch_loss.backward()
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            epoch_loss += batch_loss.item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch

        for batch in self.dataloader:

            (
                X,
                X_kept,
                targets,
                target_masks,
                padding_masks,
                padding_masks_kept,
                ids_restore,
            ) = batch

            # 1s: mask and predict, 0s: unaffected input (ignore)
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)

            # (batch_size, padded_length, feat_dim)
            if self.mae:
                X_kept = X_kept.to(self.device)
                padding_masks_kept = padding_masks_kept.to(self.device)
                ids_restore = ids_restore.to(self.device)

                predictions = self.model(
                    X_kept,
                    padding_masks,
                    padding_masks_kept,
                    ids_restore,
                )
            else:
                X = X.to(self.device)

                predictions = self.model(
                    X,
                    padding_masks,
                )

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = self.criterion(predictions, targets, target_masks)

            epoch_loss += batch_loss.item()

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        return self.epoch_metrics
