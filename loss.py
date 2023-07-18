# Reference: https://github.com/gzerveas/mvts_transformer

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F


def get_criterion(config):
    if config.task == "pretraining":
        if config.use_patch:
            return MaskedPatchLoss()
        else:
            # time series transformer operating on full signal
            return MaskedMSELoss(reduction="none")
    elif config.task == "classification":
        if config.multilabel:
            return BCEWithLogitsLoss(reduction="none")
        else:  # one class per time series
            return NoFussCrossEntropyLoss(reduction="none")
    elif config.task == "forecasting":
        if config.model_name == "hierarchical_linear":
            return HierarchicalForecastingLoss(num_levels=config.num_levels, window=2)
        return torch.nn.MSELoss(reduction="mean")
    else:
        raise ValueError(
            "Loss module for task '{}' does not exist".format(config["task"])
        )


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(
            inp,
            target.long(),  # .squeeze(),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class MaskedMSELoss(nn.Module):
    """Masked MSE Loss"""

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


class MaskedPatchLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target, mask):
        loss = (preds - target) ** 2
        # mean loss per patch
        loss = loss.mean(dim=-1)
        # mean loss on removed patches
        loss = (loss * mask).sum() / mask.sum()
        return loss


class HierarchicalForecastingLoss(nn.Module):
    def __init__(self, num_levels, window):
        super().__init__()

        self.num_levels = num_levels
        self.window = window

        enc_layers = []

        pooling_kernel = 2 ** (self.num_levels - 1)
        for _ in range(self.num_levels):
            enc_layers.append(
                nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
            )
            pooling_kernel = pooling_kernel // self.window

        self.enc_layers = nn.Sequential(*enc_layers)

        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        preds = output[0]
        preds_repeated = output[1]

        target = target.permute(0, 2, 1)
        target_features = []

        pooling_kernel = 2 ** (self.num_levels - 1)
        x = target
        for i in range(self.num_levels):
            m = self.enc_layers[i](x)
            target_features.append(m)
            m = torch.repeat_interleave(m, repeats=pooling_kernel, dim=2)
            pooling_kernel = pooling_kernel // self.window
            x = x - m

        loss = 0
        for i in range(self.num_levels):
            loss += self.loss_fn(preds[i], target_features[i])

        return loss
