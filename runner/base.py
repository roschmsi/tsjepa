# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
from collections import OrderedDict
import torch
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
