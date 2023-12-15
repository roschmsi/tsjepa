import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA
from models.patch_tst.layers.heads import (
    ClassificationPoolHead,
)


class TS2VecEMA(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """

    def __init__(
        self,
        encoder,
        predictor,
        predictor_type,
        device,
        average_top_k_layers,
        normalize_targets,
        targets_norm,
        normalize_pred,
        pred_norm,
        embed_dim,
        ema_decay,
        ema_end_decay,
        ema_anneal_end_step,
        targets_rep,
        skip_pos_embed,
        skip_patch_embed,
    ):
        super(TS2VecEMA, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.predictor = predictor
        self.predictor_type = predictor_type

        skip_keys = []

        if skip_pos_embed:
            skip_keys.append("pos_embed")
        elif skip_patch_embed:
            skip_keys.append("patch_embed.weight")
            skip_keys.append("patch_embed.bias")
        else:
            skip_keys = None

        # ema
        self.ema = EMA(
            self.encoder,
            device=device,
            ema_decay=ema_decay,
            skip_keys=skip_keys,
        )
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step

        # targets
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.targets_norm = targets_norm
        self.targets_rep = targets_rep

        self.normalize_pred = normalize_pred
        self.pred_norm = pred_norm

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

    def forward_mlp_predictor(self, X_masked):
        # model forward in online mode (student)
        # X_kept with masked tokens
        X_enc = self.encoder(X_masked)["encoder_out"]  # fetch the last layer outputs
        y_pred = self.predictor(X_enc)

        return X_enc, y_pred

    def forward_transformer_predictor(self, X_kept, ids_kept, ids_restore):
        X_enc = self.encoder(X_kept, ids_kept)["encoder_out"]
        y_pred = self.predictor(X_enc, ids_restore)

        return X_enc, y_pred

    def forward(self, X_full, X_masked, X_kept, ids_kept, ids_restore):
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
        if self.predictor_type in ["mlp", "linear"]:
            X_enc, y_pred = self.forward_mlp_predictor(X_masked=X_masked)
        elif self.predictor_type == "transformer":
            X_enc, y_pred = self.forward_transformer_predictor(
                X_kept=X_kept, ids_kept=ids_kept, ids_restore=ids_restore
            )
        else:
            raise NotImplementedError

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            if self.targets_rep in ["ffn", "block"]:
                if self.targets_rep == "ffn":
                    y = self.ema.model(X_full)["encoder_states_ffn"]
                elif self.targets_rep == "block":
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

            elif self.targets_rep == "encoder_out":
                y = self.ema.model(X_full)["encoder_out"]
                y = F.layer_norm(y.float(), y.shape[-1:])

        return X_enc, y_pred, y


class BERT(nn.Module):
    """
    Main module for masked modeling in the input space
    """

    def __init__(
        self,
        encoder,
        predictor,
        predictor_type,
        embed_dim,
    ):
        super(BERT, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.predictor_type = predictor_type
        self.embed_dim = embed_dim

    def forward_mlp_predictor(self, X_masked):
        """
        encoder and mlp predictor operate on sequence of masked and unmasked patches
        """

        X_enc = self.encoder(X_masked)["encoder_out"]
        y_pred = self.predictor(X_enc)

        return X_enc, y_pred

    def forward_transformer_predictor(self, X_kept, ids_kept, ids_restore):
        """
        encoder only operates on unmasked patches
        predictor operates on unmasked patches and mask tokens
        """

        X_enc = self.encoder(X_kept, ids_kept)["encoder_out"]
        y_pred = self.predictor(X_enc, ids_restore)

        return X_enc, y_pred

    def forward(self, X_full, X_masked, X_kept, ids_kept, ids_restore):
        """
        apply linear, MLP, or Transformer predictor
        """
        if self.predictor_type in ["mlp", "linear"]:
            X_enc, y_pred = self.forward_mlp_predictor(X_masked=X_masked)
        elif self.predictor_type == "transformer":
            X_enc, y_pred = self.forward_transformer_predictor(
                X_kept=X_kept, ids_kept=ids_kept, ids_restore=ids_restore
            )
        else:
            raise NotImplementedError

        return X_enc, y_pred, X_full


class TS2VecNoEMA(nn.Module):
    """ """

    def __init__(
        self,
        encoder,
        predictor,
        predictor_type,
        device,
        average_top_k_layers,
        normalize_targets,
        targets_rep,
        targets_norm,
        embed_dim,
    ):
        super(TS2VecNoEMA, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.predictor = predictor
        self.predictor_type = predictor_type

        # targets
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.targets_norm = targets_norm
        self.targets_rep = targets_rep

    def forward_mlp_predictor(self, X_masked):
        # model forward in online mode (student)
        X_enc = self.encoder(X_masked)["encoder_out"]  # fetch the last layer outputs
        y_pred = self.predictor(X_enc)

        return X_enc, y_pred

    def forward_transformer_predictor(self, X_kept, ids_kept, ids_restore):
        X_enc = self.encoder(X_kept, ids_kept)["encoder_out"]
        y_pred = self.predictor(X_enc, ids_restore)

        return X_enc, y_pred

    def forward(self, X_full, X_masked, X_kept, ids_kept, ids_restore):
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
        if self.predictor_type in ["mlp", "linear"]:
            X_enc, y_pred = self.forward_mlp_predictor(X_masked=X_masked)
        elif self.predictor_type == "transformer":
            X_enc, y_pred = self.forward_transformer_predictor(
                X_kept=X_kept, ids_kept=ids_kept, ids_restore=ids_restore
            )
        else:
            raise NotImplementedError

        # model forward in offline mode (teacher)
        if self.targets_rep in ["ffn", "block"]:
            if self.targets_rep == "ffn":
                y = self.encoder(X_full)["encoder_states_ffn"]
            elif self.targets_rep == "block":
                y = self.encoder(X_full)["encoder_states"]
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
            elif self.target_norm == "BatchNorm":
                y = [
                    F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)
                    for tl in y
                ]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
        elif self.targets_rep == "encoder_out":
            y = self.encoder(X_full)["encoder_out"]
            y = F.layer_norm(y.float(), y.shape[-1:])

        return X_enc, y_pred, y


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
