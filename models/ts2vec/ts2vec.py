import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class TS2Vec(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """

    def __init__(
        self,
        encoder,
        device,
        average_top_k_layers,
        normalize_targets,
        targets_norm,
        embed_dim,
        ema_decay,
        ema_end_decay,
        ema_anneal_end_step,
        skip_embeddings=False,
    ):
        super(TS2Vec, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder

        if skip_embeddings:
            skip_keys = ["pos_embed", "patch_embed.weight", "patch_embed.bias"]
        else:
            skip_keys = None

        self.ema = EMA(
            self.encoder,
            device=device,
            ema_decay=ema_decay,
            skip_keys=skip_keys,
        )  # EMA acts as the teacher
        self.regression_head = nn.Linear(self.embed_dim, self.embed_dim)

        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.targets_norm = targets_norm

        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, X_full, X_masked):
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
        # model forward in online mode (student)
        X = self.encoder(X_masked)["encoder_out"]  # fetch the last layer outputs
        # if X_full is None:
        #     return X

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(X_full)["encoder_states"]
            # fetch the last transformer layers outputs
            y = y[-self.average_top_k_layers :]
            # take the last k transformer layers

            if self.targets_norm == "LayerNorm":
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])
            elif self.targets_norm == "InstanceNorm":
                y = [
                    F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)
                    for tl in y
                ]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        X = self.regression_head(X)

        return X, y


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
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """

    def __init__(
        self,
        encoder,
        n_vars,
        d_model,
        num_patch,
        forecast_len,
        patch_len,
        head_dropout,
        head_type="linear",
    ):
        super(TS2VecForecaster, self).__init__()
        self.encoder = encoder
        self.head_type = head_type

        if head_type == "linear":
            self.head = PredictionHead(
                n_vars=n_vars,
                d_model=d_model,
                num_patch=num_patch,
                forecast_len=forecast_len,
                head_dropout=head_dropout,
            )
            self.forecast_patches = 0
        elif head_type == "transformer":
            self.head = nn.Linear(self.encoder.embed_dim, patch_len)
            self.forecast_patches = forecast_len // patch_len
        else:
            raise NotImplementedError

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

        # append 0's to mimic future values
        if self.head_type == "transformer":
            X = torch.cat(
                [
                    X,
                    torch.zeros(
                        (bs * ch, self.forecast_patches, patch_len),
                        device=X.device,
                    ),
                ],
                dim=1,
            )

        # model forward in online mode (student)
        X = self.encoder(X)["encoder_out"]  # fetch the last layer outputs
        # if X_full is None:
        #     return X

        # channel independence
        X = X.reshape(bs, ch, num_patch + self.forecast_patches, -1)

        if self.head_type == "transformer":
            X = X[:, :, -self.forecast_patches :, :]

        y = self.head(X)

        if self.head_type == "transformer":
            y = y.reshape(bs, ch, -1).transpose(1, 2)

        return y
