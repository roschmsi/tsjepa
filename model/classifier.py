import torch.nn as nn


class ClassificationPoolHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x.mean(-1)
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y


class TS2VecClassifier(nn.Module):
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
        head_type="linear",
    ):
        super(TS2VecClassifier, self).__init__()
        self.encoder = encoder

        self.head = ClassificationPoolHead(
            n_vars=n_vars,
            d_model=d_model,
            n_classes=n_classes,
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
        X = self.encoder(X)["encoder_out"]

        # channel independence
        # num_patch = num_patch + 1 if self.cls_token else num_patch
        X = X.reshape(bs, ch, num_patch, -1)
        X = X.transpose(2, 3)

        y = self.head(X)

        return y