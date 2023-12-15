import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        n_vars,
        d_model,
        num_patch,
        forecast_len,
        head_dropout=0,
        flatten=False,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, forecast_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
        x = self.dropout(x)
        x = self.linear(x)  # x: [bs x nvars x forecast_len]

        return x.transpose(1, 2)  # [bs x forecast_len x nvars]

class TS2VecForecaster(nn.Module):
    """
    Transformer model for downstream forecasting
    """

    def __init__(
        self,
        encoder,
        n_vars,
        d_model,
        num_patch,
        forecast_len,
        head_dropout,
    ):
        super(TS2VecForecaster, self).__init__()
        self.encoder = encoder

        self.head = PredictionHead(
            n_vars=n_vars,
            d_model=d_model,
            num_patch=num_patch,
            forecast_len=forecast_len,
            head_dropout=head_dropout,
        )

    def forward(self, X):
        """
        Data2Vec forward method.

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        # channel independence
        bs, num_patch, ch, patch_len = X.shape
        X = X.transpose(1, 2).reshape(bs * ch, num_patch, patch_len)

        # model forward in online mode (student)
        X = self.encoder(X)["encoder_out"]  # fetch the last layer outputs

        # channel independence
        X = X.reshape(bs, ch, num_patch, -1)

        y = self.head(X)

        return y