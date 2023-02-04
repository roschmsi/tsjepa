import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from loss import l2_reg_loss
from physionet_evaluation.evaluate_12ECG_score import compute_auc
from utils import Printer, readable_time, save_model

from sklearn.metrics import roc_auc_score

logger = logging.getLogger("__main__")
val_times = {"total_time": 0, "count": 0}


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def validate(
    val_evaluator,
    tensorboard_writer,
    config,
    best_metrics,
    best_value,
    epoch,
):
    logger.info("Evaluating on validation set ...")

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, _ = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time

    logger.info(
        "Validation runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)

    logger.info(
        "Avg validation time: {} hours, {} minutes, {} seconds".format(
            *readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch validation time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample validation time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Epoch {} Validation Summary: ".format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar("{}/val".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    # if config["key_metric"] in NEG_METRICS:
    #     condition = aggr_metrics[config["key_metric"]] < best_value
    # else:
    #     condition = aggr_metrics[config["key_metric"]] > best_value

    if aggr_metrics["loss"] < best_value:
        best_value = aggr_metrics["loss"]
        save_model(
            os.path.join(config["checkpoint_dir"], "model_best.pth"),
            epoch,
            val_evaluator.model,
        )
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


def validate_without_logging(
    val_evaluator,
    config,
    best_metrics,
    best_value,
    epoch,
):
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, _ = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1

    if aggr_metrics["loss"] < best_value:
        best_value = aggr_metrics["loss"]
        # save_model(
        #     os.path.join(config["checkpoint_dir"], "model_best.pth"),
        #     epoch,
        #     val_evaluator.model,
        # )
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


class BaseRunner(object):
    def __init__(
        self,
        model,
        dataloader,
        device,
        loss_module,
        optimizer=None,
        l2_reg=None,
        print_interval=10,
        console=True,
        multilabel=False,
        scheduler=None,
        mixup_alpha=0,
    ):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = Printer(console=console)
        self.multilabel = multilabel
        self.scheduler = scheduler

        self.epoch_metrics = OrderedDict()
        self.mixup_alpha = mixup_alpha

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError("Please override in child class")

    def evaluate(self, epoch_num=None, keep_all=True):
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
        total_active_elements = 0  # total unmasked elements in epoch

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
            loss = self.loss_module(predictions, targets, target_masks)
            batch_loss = torch.sum(loss)
            # mean loss (over active elements) used for optimization
            mean_loss = batch_loss / len(loss)

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()

        # average loss per element for whole epoch
        epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {
                "target_masks": [],
                "targets": [],
                "predictions": [],
                "metrics": [],
            }

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
            loss = self.loss_module(predictions, targets, target_masks)
            batch_loss = torch.sum(loss).cpu().item()
            # mean loss (over active elements) used for optimization the batch
            mean_loss = batch_loss / len(loss)

            if keep_all:
                per_batch["target_masks"].append(target_masks.cpu().numpy())
                per_batch["targets"].append(targets.cpu().numpy())
                per_batch["predictions"].append(predictions.cpu().numpy())
                per_batch["metrics"].append([loss.cpu().numpy()])

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss

        # average loss per element for whole epoch
        epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        lbls = []
        probs = []

        for i, batch in enumerate(self.dataloader):

            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            if self.mixup_alpha > 0:
                X, targets_a, targets_b, lam = mixup_data(
                    X, targets, self.mixup_alpha, use_cuda=True
                )

            # classification: (batch_size, num_classes) of logits

            predictions = self.model(X, padding_masks)

            # (batch_size,) loss for each sample in the batch
            # TODO make sure that loss has here dimension Nx1
            # loss = torch.mean(self.loss_module(predictions, targets), axis=1)
            if self.mixup_alpha > 0:
                loss = mixup_criterion(
                    self.loss_module, predictions, targets_a, targets_b, lam
                )
            else:
                loss = self.loss_module(predictions, targets)

            batch_loss = torch.sum(loss)
            # mean loss (over samples) used for optimization
            mean_loss = batch_loss / len(loss)

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            if torch.isnan(total_loss):
                print()

            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            self.optimizer.step()

            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

            metrics = {"loss": mean_loss.item()}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
            #     self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()

        # average loss per sample for whole epoch
        epoch_loss = epoch_loss / total_samples
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

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

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        lbls = []
        probs = []

        per_batch = {
            "target_masks": [],
            "targets": [],
            "predictions": [],
            "metrics": [],
        }

        for i, batch in enumerate(self.dataloader):

            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # classification: (batch_size, num_classes) of logits
            predictions = self.model(X, padding_masks)

            # (batch_size,) loss for each sample in the batch
            # TODO make sure that dimension is Nx1
            # loss = torch.mean(self.loss_module(predictions, targets), axis=1)
            loss = self.loss_module(predictions, targets)
            batch_loss = torch.sum(loss)
            # mean loss (over samples) used for optimization
            mean_loss = batch_loss / len(loss)

            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

            per_batch["targets"].append(targets.cpu().numpy())
            per_batch["predictions"].append(predictions.cpu().numpy())
            per_batch["metrics"].append([loss.cpu().numpy()])

            metrics = {"loss": mean_loss}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
            #     self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss

        # average loss per element for whole epoch
        epoch_loss = epoch_loss / total_samples
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)

        if self.multilabel:
            auroc, _ = compute_auc(lbls, probs)
        else:
            probs = torch.nn.functional.softmax(torch.from_numpy(probs), dim=1).numpy()
            auroc = roc_auc_score(lbls, probs, multi_class="ovo")

        self.epoch_metrics["auroc"] = auroc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class UnsupervisedPatchRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(self.dataloader):

            X, _, targets, target_masks, padding_masks, ids_restore = batch
            targets = targets.to(self.device)

            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # (batch_size, padded_length, feat_dim)
            predictions = self.model(X.to(self.device), padding_masks)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.loss_module(predictions, targets, target_masks)
            mean_loss = torch.sum(loss)
            # mean loss (over active elements) used for optimization
            # mean_loss = batch_loss / len(loss)

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                # total_active_elements += len(loss)
                epoch_loss += mean_loss.item()

        # average loss per element for whole epoch
        # epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch

        if keep_all:
            per_batch = {
                "target_masks": [],
                "targets": [],
                "predictions": [],
                "metrics": [],
            }

        for i, batch in enumerate(self.dataloader):

            X, _, targets, target_masks, padding_masks, _ = batch
            targets = targets.to(self.device)

            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # (batch_size, padded_length, feat_dim)
            predictions = self.model(X.to(self.device), padding_masks)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.loss_module(predictions, targets, target_masks)
            mean_loss = torch.sum(loss)
            # mean loss (over active elements) used for optimization
            # mean_loss = batch_loss / len(loss)

            if keep_all:
                per_batch["target_masks"].append(target_masks.cpu().numpy())
                per_batch["targets"].append(targets.cpu().numpy())
                per_batch["predictions"].append(predictions.cpu().numpy())
                per_batch["metrics"].append([loss.cpu().numpy()])

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            # total_active_elements += len(loss)
            epoch_loss += mean_loss

        # average loss per element for whole epoch
        # epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss.item()

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class UnsupervisedAERunner(BaseRunner):
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(self.dataloader):

            _, X_kept, targets, target_masks, padding_masks, ids_restore = batch
            targets = targets.to(self.device)

            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            ids_restore = ids_restore.to(self.device)

            # (batch_size, padded_length, feat_dim)
            predictions = self.model(X_kept.to(self.device), padding_masks, ids_restore)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.loss_module(predictions, targets, target_masks)
            mean_loss = torch.sum(loss) / (predictions.shape[0] * predictions.shape[2])
            # mean loss (over active elements) used for optimization
            # mean_loss = batch_loss / len(loss)

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                # total_active_elements += len(loss)
                epoch_loss += mean_loss.item()

        # average loss per element for whole epoch
        # epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch

        if keep_all:
            per_batch = {
                "target_masks": [],
                "targets": [],
                "predictions": [],
                "metrics": [],
            }

        for i, batch in enumerate(self.dataloader):

            X, X_kept, targets, target_masks, padding_masks, ids_restore = batch
            targets = targets.to(self.device)

            # 1s: mask and predict, 0s: unaffected input (ignore)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            ids_restore = ids_restore.to(self.device)

            # (batch_size, padded_length, feat_dim)
            predictions = self.model(X_kept.to(self.device), padding_masks, ids_restore)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            # (num_active,) individual loss (square error per element) for each active value in batch
            loss = self.loss_module(predictions, targets, target_masks)
            mean_loss = torch.sum(loss) / (predictions.shape[0] * predictions.shape[2])
            # mean loss (over active elements) used for optimization
            # mean_loss = batch_loss / len(loss)

            if keep_all:
                per_batch["target_masks"].append(target_masks.cpu().numpy())
                per_batch["targets"].append(targets.cpu().numpy())
                per_batch["predictions"].append(predictions.cpu().numpy())
                per_batch["metrics"].append([loss.cpu().numpy()])

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            # total_active_elements += len(loss)
            epoch_loss += mean_loss

        # average loss per element for whole epoch
        # epoch_loss = epoch_loss / total_active_elements
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
