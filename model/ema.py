# Reference: 

import os
import copy

import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMAModule.

    Args:
        model (nn.Module):
        device (str):
        skip_keys (list): The keys to skip assigning averaged weights to.
    """

    def __init__(self, model: nn.Module, device, ema_decay, skip_keys=None):
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.device = device
        self.model.to(self.device)
        self.skip_keys = skip_keys or set()
        self.decay = ema_decay
        self.num_updates = 0

    def step(self, new_model: nn.Module):
        """
        One EMA step

        Args:
            new_model (nn.Module): Online model to fetch new weights from

        """
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)

        self.num_updates += 1

    @staticmethod
    def get_annealed_rate(start, end, curr_step, total_steps):
        """
        Calculate EMA annealing rate
        """
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining
