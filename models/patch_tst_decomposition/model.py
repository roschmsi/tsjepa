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
from models.patch_tst.layers.revin import RevIN
from data.dataset import create_patch


class PatchTSTDecomposition(nn.Module):
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
        separate_backbone=False,
        trend_seasonal_residual=False,
    ):
        super().__init__()

        self.revin = revin
        self.d_model = d_model
        self.num_patch = num_patch
        self.separate_backbone = separate_backbone
        self.trend_seasonal_residual = trend_seasonal_residual

        # TODO fix this, if not channel independence
        if self.revin:
            self.revin_trend = RevIN(num_features=1, affine=True, subtract_last=False)
            if self.trend_seasonal_residual:
                self.revin_seasonal = RevIN(
                    num_features=1, affine=True, subtract_last=False
                )
            self.revin_residual = RevIN(
                num_features=1, affine=True, subtract_last=False
            )

        self.patch_len = patch_len

        if separate_backbone:
            self.backbone_trend = PatchTSTDecompositionEncoder(
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
            if self.trend_seasonal_residual:
                self.backbone_seasonal = PatchTSTDecompositionEncoder(
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
            self.backbone_residual = PatchTSTDecompositionEncoder(
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
        else:
            self.backbone = PatchTSTDecompositionEncoder(
                c_in=c_in,
                num_patch=num_patch * 3,  # TODO fix this, residual + seasonal + trend
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

        if task == "forecasting":
            self.head_trend = DecompositionPredictionHead(
                n_vars=c_in, d_model=d_model, num_patch=num_patch, forecast_len=c_out
            )
            if self.trend_seasonal_residual:
                self.head_seasonal = DecompositionPredictionHead(
                    n_vars=c_in,
                    d_model=d_model,
                    num_patch=num_patch,
                    forecast_len=c_out,
                )
            self.head_residual = DecompositionPredictionHead(
                n_vars=c_in, d_model=d_model, num_patch=num_patch, forecast_len=c_out
            )
        else:
            raise ValueError(f"Task {task} not defined.")

    def forward(
        self,
        trend,
        seasonal,
        residual,
        padding_mask=None,
        X_time=None,
        y_time=None,
        return_encoding=False,
    ):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        # bs x seq_len x n_vars
        bs, seq_len, n_vars = trend.shape

        # channel independence
        trend = trend.transpose(1, 2).reshape(bs * n_vars, seq_len).unsqueeze(-1)
        if self.trend_seasonal_residual:
            seasonal = (
                seasonal.transpose(1, 2).reshape(bs * n_vars, seq_len).unsqueeze(-1)
            )
        residual = residual.transpose(1, 2).reshape(bs * n_vars, seq_len).unsqueeze(-1)

        if self.revin:
            trend = self.revin_trend(trend, mode="norm")
            if self.trend_seasonal_residual:
                seasonal = self.revin_seasonal(seasonal, mode="norm")
            residual = self.revin_residual(residual, mode="norm")

        # patching
        trend = create_patch(trend, patch_len=self.patch_len, stride=self.patch_len)
        if self.trend_seasonal_residual:
            seasonal = create_patch(
                seasonal, patch_len=self.patch_len, stride=self.patch_len
            )
        residual = create_patch(
            residual, patch_len=self.patch_len, stride=self.patch_len
        )

        if self.separate_backbone:
            trend = self.backbone_trend(trend)
            if self.trend_seasonal_residual:
                seasonal = self.backbone_seasonal(seasonal)
            residual = self.backbone_residual(residual)
        else:
            if self.trend_seasonal_residual:
                X = torch.cat([trend, seasonal, residual], dim=1)
                z = self.backbone(X)
                trend, seasonal, residual = torch.split(z, self.num_patch, dim=2)
            else:
                X = torch.cat([trend, residual], dim=1)
                z = self.backbone(X)
                trend, residual = torch.split(z, self.num_patch, dim=2)

        # bs * n_vars x num_patch x 1 x d_model

        trend = self.head_trend(trend)
        if self.trend_seasonal_residual:
            seasonal = self.head_seasonal(seasonal)
        residual = self.head_residual(residual)

        if self.revin:
            trend = self.revin_trend(trend, mode="denorm")
            if self.trend_seasonal_residual:
                seasonal = self.revin_seasonal(seasonal, mode="denorm")
            residual = self.revin_residual(residual, mode="denorm")

        if self.trend_seasonal_residual:
            pred = trend + seasonal + residual
        else:
            pred = trend + residual

        # channel independence
        _, pred_len, _ = pred.shape
        pred = pred.reshape(bs, n_vars, pred_len).transpose(1, 2)

        return pred


class PatchTSTDecompositionEncoder(nn.Module):
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
        # x = self.dropout(x + self.W_pos)
        x = x + self.W_pos

        # apply transformer encoder
        z = self.encoder(x)

        z = z.reshape(bs, n_vars, num_patch, self.d_model)
        # z: [bs x nvars x num_patch x d_model]

        return z


class DecompositionPredictionHead(nn.Module):
    def __init__(
        self,
        n_vars,
        d_model,
        num_patch,
        forecast_len,
    ):
        super().__init__()
        self.n_vars = n_vars
        head_dim = d_model * num_patch

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, forecast_len)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
        x = self.linear(x)  # x: [bs x nvars x forecast_len]
        x = x.transpose(1, 2)  # x: [bs x forecast_len x nvars]

        return x
