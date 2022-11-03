import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


def get_loss(config):
    if config["task"] == "imputation":
        return MaskedMSELoss(reduction="none")
    elif config["task"] == "classification":
        return BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError(
            "Loss module for task '{}' does not exist".format(config["task"])
        )


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == "output_layer.weight":
            return torch.sum(torch.square(param))


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
