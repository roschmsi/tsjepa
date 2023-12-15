import torch.nn as nn


class PoolingHead(nn.Module):
    def __init__(
        self,
        n_vars,
        d_model,
        n_classes,
        head_dropout=0,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        x = x.mean(-1)
        x = self.flatten(x)
        x = self.dropout(x)

        y = self.linear(x)

        return y


class Classifier(nn.Module):
    """
    Transformer model for downstream classification
    """

    def __init__(
        self,
        encoder,
        n_vars,
        d_model,
        n_classes,
        head_dropout,
    ):
        super(Classifier, self).__init__()
        self.encoder = encoder

        self.head = PoolingHead(
            n_vars=n_vars,
            d_model=d_model,
            n_classes=n_classes,
            head_dropout=head_dropout,
        )

    def forward(self, X):
        # channel independence
        bs, num_patch, ch, patch_len = X.shape
        X = X.transpose(1, 2).reshape(bs * ch, num_patch, patch_len)

        X = self.encoder(X)["encoder_out"]

        # channel independence
        X = X.reshape(bs, ch, num_patch, -1)
        X = X.transpose(2, 3)

        y = self.head(X)

        return y
