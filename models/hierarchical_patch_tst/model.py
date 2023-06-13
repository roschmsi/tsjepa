# Reference: https://github.com/yuqinie98/PatchTST

import math
import torch
from torch import Tensor, nn

from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.heads import (
    ClassificationHead,
)
from models.patch_tst.layers.pos_encoding import positional_encoding


class PretrainHead(nn.Module):
    def __init__(self, n_levels, n_vars, d_model, num_patch, patch_len, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)
        self.patch_len = patch_len

        modules = []
        n_len = patch_len
        for n in range(n_levels):
            modules.append(nn.Linear(d_model, n_len))
            n_len = math.ceil(n_len * 2)

        self.mlp = nn.ModuleList(modules)

        self.mix = nn.Linear(n_levels * patch_len, patch_len)

    def forward(self, x_enc):
        """
        x: tensor [bs x nvars x num_patch x d_model]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        predicted = []

        for i in range(len(x_enc)):
            bs, n_vars, num_patch, d_model = x_enc[i].shape
            x = x_enc[i]  # [bs x nvars x num_patch x d_model]
            x = self.mlp[i](self.dropout(x))  # [bs x nvars x num_patch x patch_len]
            x = x.reshape(bs, n_vars, -1, self.patch_len)
            predicted.append(x)

        x = self.mix(torch.cat(predicted, -1))

        x = x.transpose(1, 2)  # [bs x num_patch x nvars x patch_len]
        return x


class PredictionHead(nn.Module):
    def __init__(
        self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        modules = []
        n_len = num_patch
        for n in range(n_levels):
            modules.append(nn.Linear(n_len * d_model, forecast_len))
            n_len = math.ceil(n_len / 2)

        self.mlp = nn.ModuleList(modules)

        self.mix = nn.Linear(n_levels * forecast_len, forecast_len)

    def forward(self, x_enc):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        # bs, num_patch, d_model = x.shape

        forecasted = []

        for i in range(len(x_enc)):
            x = self.flatten(x_enc[i])
            # x = self.dropout(x)
            x = self.mlp[i](x)
            forecasted.append(x)

        # y = torch.stack(forecasted, dim=-1).sum(-1)
        y = torch.cat(forecasted, dim=-1)
        y = self.mix(y)

        y = y.transpose(1, 2)

        return y


class ResidualPredictionHead(nn.Module):
    def __init__(
        self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        modules = []
        look_back_len = num_patch
        horizon_len = forecast_len
        for _ in range(n_levels):
            modules.append(nn.Linear(look_back_len * d_model, horizon_len))
            look_back_len = math.ceil(look_back_len / 2)
            horizon_len = math.ceil(horizon_len / 2)

        self.mlp = nn.ModuleList(modules)

        # self.mix = nn.Linear(n_levels * forecast_len, forecast_len)

    def forward(self, x_enc):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        # bs, num_patch, d_model = x.shape

        forecasted = []

        for i in range(len(x_enc)):
            x = self.flatten(x_enc[i])
            # x = self.dropout(x)
            x = self.mlp[i](x)
            x = torch.repeat_interleave(x, repeats=(2**i), dim=2)
            forecasted.append(x)

        y = torch.stack(forecasted).sum(0)

        y = y.transpose(1, 2)

        return y


class LowResPredictionHead(nn.Module):
    def __init__(
        self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        n_len = math.ceil(num_patch / 2 ** (n_levels - 1))
        self.mlp = nn.Linear(n_len * d_model, forecast_len)

    def forward(self, x_enc):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        # bs, num_patch, d_model = x.shape
        x = self.flatten(x_enc[-1])
        y = self.mlp(x)
        y = y.transpose(1, 2)

        return y


class DownsamplingMLP(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: bs, num_patch, d_model
        """
        batch_size, num_patch, d_model = x.shape
        pad_num = num_patch % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, -pad_num:, :]), dim=1)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, i :: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        # x = self.norm(x)
        x = self.linear_trans(x)

        return x


class UpsamplingMLP(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(d_model, win_size * d_model)
        self.norm = norm_layer(win_size * d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: bs, num_patch, d_model
        """
        batch_size, num_patch, d_model = x.shape

        x = self.linear_trans(x)
        x = x.reshape(batch_size, -1)
        x = x.reshape(batch_size, -1, d_model)

        return x


class HierarchicalPatchTST(nn.Module):
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
        num_levels: int,
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

        self.backbone = HierarchicalPatchTSTEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            num_levels=num_levels,
            num_layers=2 * num_layers,
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
            # self.head = PretrainUNetHead(
            #     n_levels=num_levels,
            #     n_vars=c_in,
            #     d_model=d_model,
            #     num_patch=num_patch,
            #     patch_len=patch_len,
            #     head_dropout=head_dropout,
            # )
            self.head = HierarchicalPatchTSTDecoder(
                c_in=c_in,
                num_patch=num_patch,
                patch_len=patch_len,
                num_levels=num_levels,
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
        elif task == "forecasting":
            self.head = ResidualPredictionHead(
                n_levels=num_levels,
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
        z = self.backbone(z)
        # z: [bs x nvars x d_model x num_patch]
        z = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class HierarchicalPatchTSTEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        num_levels,
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

        # self.W_pos = []
        # pos_enc_len = num_patch

        # for i in range(num_levels):
        #     self.W_pos.append(positional_encoding(pe, learn_pe, pos_enc_len, d_model))
        #     pos_enc_len = math.ceil(pos_enc_len / 2)

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        enc_modules = []

        for i in range(num_levels - 1):
            # encoder
            encoder = TSTEncoder(
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
            mlp = DownsamplingMLP(
                d_model=d_model, win_size=2, norm_layer=nn.BatchNorm1d
            )
            enc_modules.append(nn.Sequential(encoder, mlp))

        enc_modules.append(
            nn.Sequential(
                TSTEncoder(
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
            )
        )

        self.encoder = nn.Sequential(*enc_modules)

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
        # x = x.reshape(bs, n_vars, num_patch, self.d_model)

        # apply transformer encoder
        # z = self.encoder(x)
        x_enc = []

        for i, (name, module) in enumerate(self.encoder.named_children()):
            for name, m_layer in module.named_children():
                # pos encoding
                # x = self.dropout(x + self.W_pos[i].cuda())
                x = m_layer(x)

                if type(m_layer) is TSTEncoder:
                    x_enc.append(x.reshape(bs, n_vars, -1, self.d_model))

        # z = z.reshape(bs, n_vars, -1, self.d_model)
        # # z: [bs x nvars x num_patch x d_model]

        # # prepare output, remove class and channel token
        # if self.ch_token is not None:
        #     z = z[:, :, :-1, :]
        # if self.task != "classification" and self.cls_token is not None:
        #     z = z[:, :, 1:, :]

        # z = z.transpose(2, 3)
        # # z: [bs x nvars x d_model x num_patch]

        return x_enc


class HierarchicalPatchTSTDecoder(nn.Module):
    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        num_levels,
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

        # self.W_pos = []
        # pos_enc_len = num_patch

        # for i in range(num_levels):
        #     self.W_pos.append(positional_encoding(pe, learn_pe, pos_enc_len, d_model))
        #     pos_enc_len = math.ceil(pos_enc_len / 2)

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        dec_modules = []

        # encoder
        encoder = TSTEncoder(
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
        dec_modules.append(nn.Sequential(encoder))

        for i in range(num_levels - 1):
            dec_modules.append(
                nn.Sequential(
                    UpsamplingMLP(
                        d_model=d_model, win_size=2, norm_layer=nn.BatchNorm1d
                    ),
                    nn.Linear(2 * d_model, d_model),
                    TSTEncoder(
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
                    ),
                )
            )

        self.decoder = nn.Sequential(*dec_modules)

        self.final_proj = nn.Linear(d_model, patch_len)

    def forward(self, x_enc) -> Tensor:
        y = x_enc[-1]

        y_dec = []

        bs, n_vars, num_patch, patch_len = y.shape

        # x: [bs x nvars x num_patch x d_model]
        y = y.reshape(bs * n_vars, -1, self.d_model)
        # x: [bs * nvars x num_patch x d_model]

        for i, (name, module) in enumerate(self.decoder.named_children()):
            for name, m_layer in module.named_children():
                # concatenate encoder outputs
                if type(m_layer) is nn.Linear:
                    x = x_enc[-i - 1]
                    x = x.reshape(bs * n_vars, -1, self.d_model)
                    y = torch.cat([y, x], dim=2)

                y = m_layer(y)

                if type(m_layer) is TSTEncoder:
                    y_dec.append(
                        y.reshape(bs, n_vars, -1, self.d_model).transpose(1, 2)
                    )

        pred = self.final_proj(y_dec[-1])

        return pred
