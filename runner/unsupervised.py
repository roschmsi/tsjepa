# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import torch

from runner.base import BaseRunner

logger = logging.getLogger("__main__")


class UnsupervisedRunner(BaseRunner):
    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()

        epoch_loss = 0

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
                    target_masks,
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
                    X_kept, padding_masks, padding_masks_kept, ids_restore, target_masks
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
