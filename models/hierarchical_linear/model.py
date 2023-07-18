import torch
import torch.nn as nn


class HierarchialLinear(nn.Module):
    """
    Just a few Linear layer
    """

    def __init__(self, seq_len=512, pred_len=96, enc_in=7, window=2, num_levels=4):
        super(HierarchialLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
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

        dec_layers = []

        look_back = self.seq_len
        horizon = self.pred_len

        for _ in range(self.num_levels):
            dec_layers.append(
                nn.Sequential(
                    nn.Linear(look_back, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, horizon),
                )
            )
            look_back = look_back // self.window
            horizon = horizon // self.window

        dec_layers = dec_layers[::-1]
        self.dec_layers = nn.Sequential(*dec_layers)

        # save the meaned values per level or only the residuals?
        # first try: residuals only

    def forward(self, x, padding_mask=None):
        # parameter-free encoder
        # x: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)

        enc_features = []

        pooling_kernel = 2 ** (self.num_levels - 1)
        for i in range(self.num_levels):
            m = self.enc_layers[i](x)
            enc_features.append(m)
            m = torch.repeat_interleave(m, repeats=pooling_kernel, dim=2)
            pooling_kernel = pooling_kernel // self.window
            x = x - m

        assert x.sum().item() == 0

        preds = []
        preds_repeated = []

        repeats = 2 ** (self.num_levels - 1)

        for i in range(self.num_levels):
            pred = self.dec_layers[i](enc_features[i])
            preds.append(pred)

            pred_rep = torch.repeat_interleave(pred, repeats, dim=2)
            preds_repeated.append(pred_rep)
            repeats = repeats // self.window

        return preds, preds_repeated
