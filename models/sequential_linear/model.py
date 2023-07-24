import torch
import torch.nn as nn


class SeqLinear(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(SeqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Use this line if you want to visualize the weights
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, 1))
        else:
            self.Linear = nn.Linear(self.seq_len, 1)

    def forward(self, x, padding_mask=None):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            for i in range(self.pred_len):
                pred = self.Linear(x[:, -self.seq_len :, :].permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                x = torch.cat([x, pred], dim=1)
        return x[:, -self.pred_len :, :]  # [Batch, Output length, Channel]
