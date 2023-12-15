import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        d_model,
        num_patch,
        forecast_len,
        head_dropout=0,
    ):
        super().__init__()
        head_dim = d_model * num_patch

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, forecast_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)

        y = self.linear(x)

        return y.transpose(1, 2)


class Forecaster(nn.Module):
    """
    Transformer model for downstream forecasting
    """

    def __init__(
        self,
        encoder,
        d_model,
        num_patch,
        forecast_len,
        head_dropout,
    ):
        super(Forecaster, self).__init__()
        self.encoder = encoder

        self.head = PredictionHead(
            d_model=d_model,
            num_patch=num_patch,
            forecast_len=forecast_len,
            head_dropout=head_dropout,
        )

    def forward(self, X):
        # channel independence
        bs, num_patch, ch, patch_len = X.shape
        X = X.transpose(1, 2).reshape(bs * ch, num_patch, patch_len)

        X = self.encoder(X)["encoder_out"]

        # channel independence
        X = X.reshape(bs, ch, num_patch, -1)

        y = self.head(X)

        return y
