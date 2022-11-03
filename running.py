import logging
import os
from collections import OrderedDict
import time

import torch
import numpy as np
from loss import l2_reg_loss
import utils
from physionet_evaluation.evaluate_12ECG_score import (
    compute_auc,
    compute_challenge_metric,
)


logger = logging.getLogger("__main__")
NEG_METRICS = {"loss"}  # metrics for which "better" is less
val_times = {"total_time": 0, "count": 0}


def validate(
    val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch
):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time

    logger.info(
        "Validation runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)

    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Epoch {} Validation Summary: ".format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar("{}/val".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    if config["key_metric"] in NEG_METRICS:
        condition = aggr_metrics[config["key_metric"]] < best_value
    else:
        condition = aggr_metrics[config["key_metric"]] > best_value

    if condition:
        best_value = aggr_metrics[config["key_metric"]]
        utils.save_model(
            os.path.join(config["checkpoint_dir"], "model_best.pth"),
            epoch,
            val_evaluator.model,
        )
        best_metrics = aggr_metrics.copy()

        # pred_filepath = os.path.join(config["output_dir"], "best_predictions")
        # np.savez(pred_filepath, **per_batch)

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
    ):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

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

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(
                self.device
            )  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(
                X.to(self.device), padding_masks
            )  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(
                predictions, targets, target_masks
            )  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(
                loss
            )  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_active_elements
        )  # average loss per element for whole epoch
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
            target_masks = target_masks.to(
                self.device
            )  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(
                X.to(self.device), padding_masks
            )  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(
                predictions, targets, target_masks
            )  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(
                loss
            )  # mean loss (over active elements) used for optimization the batch

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
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_active_elements
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(SupervisedRunner, self).__init__(*args, **kwargs)

        self.classification = True  # True if classification, False if regression

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

            # classification: (batch_size, num_classes) of logits
            predictions = self.model(X, padding_masks)

            loss = torch.sum(
                self.loss_module(predictions, targets), axis=1
            )  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(
                loss
            )  # mean loss (over samples) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)
        auroc, auprc = compute_auc(lbls, probs)
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

            loss = torch.sum(
                self.loss_module(predictions, targets), axis=1
            )  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

            per_batch["targets"].append(targets.cpu().numpy())
            per_batch["predictions"].append(predictions.cpu().numpy())
            per_batch["metrics"].append([loss.cpu().numpy()])

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)
        auroc, auprc = compute_auc(lbls, probs)
        self.epoch_metrics["auroc"] = auroc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
