# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import numpy as np
import torch
from data.mixup import mixup_criterion, mixup_data
from evaluation.evaluate_12ECG_score import compute_auc
from runner.base import BaseRunner

logger = logging.getLogger("__main__")


class SupervisedRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):
        self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        lbls = []
        probs = []

        for batch in self.dataloader:
            X, targets, padding_masks = batch
            X = X.to(self.device)
            targets = targets.to(self.device).float()
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

        if self.scheduler is not None:
            self.scheduler.step()

        if self.multilabel:
            auroc, _ = compute_auc(lbls, probs)
        else:
            probs = torch.nn.functional.softmax(torch.from_numpy(probs), dim=1).numpy()
            auroc = 0
            # auroc = roc_auc_score(lbls, probs, multi_class="ovo")
            # TODO reactivate AUROC

        self.epoch_metrics["auroc"] = auroc

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        lbls = []
        probs = []

        for batch in self.dataloader:
            X, targets, padding_masks = batch
            X = X.to(self.device)
            # TODO better conversion to float
            targets = targets.to(self.device).float()
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
            # auroc = roc_auc_score(lbls, probs, multi_class="ovo")
            auroc = 0
            # TODO reactivate AUROC

        self.epoch_metrics["auroc"] = auroc

        return self.epoch_metrics
