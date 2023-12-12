# Reference: https://github.com/yuqinie98/PatchTST

import torch
from torch import Tensor, nn

from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.heads import (
    ClassificationTokenHead,
    PredictionHead,
    PatchRevinHead,
    PretrainHead,
)
from models.patch_tst.layers.pos_encoding import positional_encoding
from models.patch_tst.layers.revin import RevIN


class PatchTST(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_patch: int,
        patch_len: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        revin: bool = False,
        patch_revin: bool = False,
        shared_embedding=True,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token=False,
        ch_token=False,
        attn_dropout: float = 0.0,
        res_attention: bool = False,
        store_attn: bool = False,
        task=None,
        head_dropout=0,
        individual=False,
    ):
        super().__init__()

        self.revin = revin
        self.patch_revin = patch_revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.encoder = TransformerEncoder(
            seq_len=max_seq_len,
            num_patch=num_patch,
            patch_size=config.patch_len,
            in_chans=config.feat_dim,
            embed_dim=config.enc_d_model,
            depth=config.enc_num_layers,
            num_heads=config.enc_num_heads,
            mlp_ratio=config.enc_mlp_ratio,
            drop_rate=config.dropout,
            attn_drop_rate=config.attn_drop_rate,
            activation=config.activation,
            activation_drop_rate=config.activation_drop_rate,
            norm=config.norm,
            layer_norm_first=config.layer_norm_first,
            learn_pe=config.learn_pe,
        )

        self.predictor = 

        if task == "pretraining":
            self.head = PretrainHead(
                d_model=d_model,
                patch_len=patch_len,
                head_dropout=head_dropout,
            )
        elif task == "forecasting":
            if self.patch_revin:
                self.head = PatchRevinHead(
                    n_vars=c_in,
                    d_model=d_model,
                    input_num_patch=num_patch,
                    output_num_patch=c_out // patch_len,
                    forecast_len=c_out,
                    head_dropout=head_dropout,
                    patch_len=patch_len,
                )
            else:
                self.head = PredictionHead(
                    individual=individual,
                    n_vars=c_in,
                    d_model=d_model,
                    num_patch=num_patch,
                    forecast_len=c_out,
                    head_dropout=head_dropout,
                )
        elif task == "classification":
            self.head = ClassificationTokenHead(
                n_vars=c_in,
                d_model=d_model,
                n_patch=num_patch,
                n_classes=c_out,
                head_dropout=head_dropout,
            )
        else:
            raise ValueError(f"Task {task} not defined.")

    def forward(
        self, z, padding_mask=None, X_time=None, y_time=None, return_encoding=False
    ):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = z.shape

        # bs x nvars x seq_len
        if self.revin:
            z = z.transpose(1, 2)
            z = z.reshape(bs, n_vars, num_patch * patch_len)  # bs x n_vars x seq_len
            z = z.transpose(1, 2)  # bs x seq_len x n_vars

            z = self.revin_layer(z, mode="norm")

            z = z.transpose(1, 2)
            z = z.reshape(bs, n_vars, num_patch, patch_len)
            z = z.transpose(1, 2)
        elif self.patch_revin:
            mean = z.mean(dim=-1, keepdim=True)
            std = torch.sqrt(z.var(dim=-1, keepdim=True) + 1e-4)
            z = (z - mean) / std

        # TODO do not use float here, ensure datatype in dataset class
        z = self.backbone(z)
        # z: [bs x nvars x d_model x num_patch]

        if self.patch_revin:
            pred = self.head(z, mean, std)
        else:
            pred = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain

        if self.revin:
            # pred = pred.permute(0, 2, 1)
            pred = self.revin_layer(pred, "denorm")
            # pred = pred.permute(0, 2, 1)

        if return_encoding:
            return pred, z
        else:
            return pred
