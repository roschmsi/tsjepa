# Reference: https://github.com/gzerveas/mvts_transformer

import logging
from collections import OrderedDict

from utils.setup import Printer

logger = logging.getLogger("__main__")

class BaseRunner(object):
    def __init__(
        self,
        model,
        dataloader,
        device,
        criterion,
        optimizer=None,
        scheduler=None,
        print_interval=10,
        console=True,
        multilabel=False,
        mixup=0,
        use_time_features=False,
        layer_wise_prediction=False,
        hierarchical_loss=False,
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
        self.use_time_features = use_time_features
        self.layer_wise_prediction = layer_wise_prediction
        self.hierarchical_loss = hierarchical_loss

        self.epoch_metrics = OrderedDict()
        self.mixup = mixup
        self.vic_reg = False
        self.vibc_reg = False
        self.vic_reg_enc = False

    def train_epoch(self):
        raise NotImplementedError("Please override in child class")

    def evaluate(self):
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
