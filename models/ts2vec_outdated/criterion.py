import logging

import torch
from torch.nn.modules.loss import _Loss

logger = logging.getLogger(__name__)


class TS2VecCriterion(_Loss):
    def __init__(self, loss_weights=None, log_keys=None, can_sum=True):
        super().__init__()
        self.loss_weights = loss_weights
        self.log_keys = log_keys
        self.can_sum = can_sum

    def forward(self, model, sample, reduce=True):
        net_output = model(
            sample["net_input"]["imgs"].cuda(),
            precomputed_mask=sample["net_input"]["precomputed_mask"].cuda(),
        )

        scaled_losses = {}

        losses = net_output["losses"]

        for lk, p in losses.items():
            try:
                coef = 1.0 if len(self.loss_weights) == 0 else self.loss_weights[lk]
            except KeyError:
                logger.error(
                    f"weight for loss {lk} is not in loss_weights ({self.loss_weights})"
                )
                raise
            if coef != 0 and p is not None:
                scaled_losses[lk] = coef * p.float().sum()

        loss = sum(scaled_losses.values())

        if "sample_size" in net_output:
            sample_size = net_output["sample_size"]
        else:
            sample_size = loss.numel()

        if reduce and loss.numel() > 1:
            loss = loss.sum()

        # logging

        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "_world_size": 1,
        }

        for lk in self.log_keys:
            if lk in net_output and net_output[lk] is not None:
                if not torch.is_tensor(net_output[lk]) or net_output[lk].numel() == 1:
                    logging_output[lk] = float(net_output[lk])
                elif lk.startswith("_"):
                    logging_output[lk] = net_output[lk]
                else:
                    for i, v in enumerate(net_output[lk]):
                        logging_output[f"{lk}_{i}"] = float(v)

        if len(scaled_losses) > 1:
            for lk, l in scaled_losses.items():
                if l.numel() > 1:
                    l = l.sum()
                logging_output[f"d{lk}"] = l.item()

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        return loss, sample_size, logging_output
