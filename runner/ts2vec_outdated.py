# Reference: https://github.com/gzerveas/mvts_transformer

import logging
from runner.base import BaseRunner

logger = logging.getLogger("__main__")


class TS2VecRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_num = 0

    def train_epoch(self, epoch_num=None):
        self.model.train()
        self.model.set_num_updates(self.update_num)  # set updated num

        epoch_loss = 0
        target_var = 0
        pred_var = 0

        for batch in self.dataloader:
            # X, targets, padding_masks = batch
            # X = X.to(self.device)
            # targets = targets.to(self.device)
            # padding_masks = padding_masks.to(self.device)  # 0s: ignore

            loss, sample_size, logging_output = self.criterion(self.model, batch)
            loss = loss / sample_size  # average loss per element
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            target_var += logging_output["target_var"]
            pred_var += logging_output["pred_var"]

            self.update_num += 1

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        self.epoch_metrics["target_var"] = target_var / len(self.dataloader)
        self.epoch_metrics["pred_var"] = pred_var / len(self.dataloader)

        return self.epoch_metrics

    def evaluate(self, epoch_num=None):
        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        target_var = 0
        pred_var = 0

        for batch in self.dataloader:
            loss, sample_size, logging_output = self.criterion(self.model, batch)
            loss = loss / sample_size

            epoch_loss += loss.item()

            target_var += logging_output["target_var"]
            pred_var += logging_output["pred_var"]

        # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss / len(self.dataloader)

        self.epoch_metrics["target_var"] = target_var / len(self.dataloader)
        self.epoch_metrics["pred_var"] = pred_var / len(self.dataloader)

        return self.epoch_metrics
