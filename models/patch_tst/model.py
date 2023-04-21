# Reference: https://github.com/yuqinie98/PatchTST

import torch
from torch import Tensor, nn

from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.heads import (
    ClassificationHead,
    PredictionHead,
    PretrainHead,
)
from models.patch_tst.layers.pos_encoding import positional_encoding


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
        shared_embedding=True,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token=False,
        ch_token=False,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
        store_attn: bool = False,
        task=None,
        head_dropout=0,
        individual=False,
    ):
        super().__init__()

        self.backbone = PatchTSTEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            shared_embedding=shared_embedding,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            pe=pe,
            learn_pe=learn_pe,
            cls_token=cls_token,
            ch_token=ch_token,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
            store_attn=store_attn,
            task=task,
        )

        if task == "pretraining":
            self.head = PretrainHead(
                d_model=d_model,
                patch_len=patch_len,
                head_dropout=head_dropout,
            )
        elif task == "forecasting":
            self.head = PredictionHead(
                individual=individual,
                n_vars=c_in,
                d_model=d_model,
                num_patch=num_patch,
                forecast_len=c_out,
                head_dropout=head_dropout,
            )
        elif task == "classification":
            self.head = ClassificationHead(
                n_vars=c_in,
                d_model=d_model,
                n_patch=num_patch,
                n_classes=c_out,
                head_dropout=head_dropout,
            )
        else:
            raise ValueError(f"Task {task} not defined.")

    def forward(self, z, padding_mask=None):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        z = self.backbone(z.float())
        # z: [bs x nvars x d_model x num_patch]
        z = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        shared_embedding=True,
        norm="BatchNorm",
        pre_norm=False,
        activation="gelu",
        pe="zeros",
        learn_pe=False,
        cls_token=False,
        ch_token=False,
        attn_dropout=0.0,
        store_attn=False,
        res_attention=True,
        task=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.task = task

        # input encoding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(c_in):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # class and channel tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if cls_token else None
        self.ch_token = (
            nn.Parameter(torch.zeros(c_in, 1, d_model)) if ch_token else None
        )

        # positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = TSTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
            store_attn=store_attn,
        )

    def forward(self, x) -> Tensor:
        bs, num_patch, n_vars, patch_len = x.shape

        # input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)

        # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1, 2)
        # x: [bs x nvars x num_patch x d_model]
        x = x.reshape(bs * n_vars, num_patch, self.d_model)
        # x: [bs * nvars x num_patch x d_model]

        # add positional encoding
        x = self.dropout(x + self.W_pos)

        x = x.reshape(bs, n_vars, num_patch, self.d_model)

        # append channel and class token
        if self.ch_token is not None:
            ch_token = self.ch_token.expand(bs, -1, -1, -1)
            x = torch.cat((x, ch_token), dim=2)
            num_patch += 1
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(bs, n_vars, -1, -1)
            x = torch.cat((cls_token, x), dim=2)
            num_patch += 1

        x = x.reshape(bs * n_vars, num_patch, self.d_model)

        # apply transformer encoder
        z = self.encoder(x)

        z = z.reshape(bs, n_vars, num_patch, self.d_model)
        # z: [bs x nvars x num_patch x d_model]

        # prepare output, remove class and channel token
        if self.ch_token is not None:
            z = z[:, :, :-1, :]
        if self.task != "classification" and self.cls_token is not None:
            z = z[:, :, 1:, :]

        z = z.transpose(2, 3)
        # z: [bs x nvars x d_model x num_patch]

        return z
