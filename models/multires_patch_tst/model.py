# Reference: https://github.com/yuqinie98/PatchTST

import math
import torch
from torch import Tensor, nn
from models.hierarchical_patch_tst.mlp import DownsamplingMLP, UpsamplingMLP

from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.pos_encoding import positional_encoding
from models.hierarchical_patch_tst.decoder import TSTDecoder
from models.patch_tst.layers.revin import RevIN
from data.dataset import create_patch


class MultiresPatchTST(nn.Module):
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
        max_seq_len: int,
        num_levels: int,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
        dec_num_layers: int,
        dec_num_heads: int,
        dec_d_model: int,
        dec_d_ff: int,
        window_size: list,
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
        use_time_features=False,
        layer_wise_prediction=False,
        revin=False,
        interpolation=False,
    ):
        super().__init__()
        self.c_in = c_in
        self.layer_wise_prediction = layer_wise_prediction
        self.revin = revin
        self.max_seq_len = max_seq_len
        self.interpolation = interpolation
        self.num_levels = num_levels
        self.pred_len = c_out

        if self.revin:
            self.revin_layer = RevIN(
                num_features=c_in, affine=True, subtract_last=False
            )

        self.enc_num_patches = []
        for i in range(num_levels):
            enc_num_patch = math.floor(max_seq_len / window_size[i])
            self.enc_num_patches.append(enc_num_patch)

        self.dec_num_patches = []
        for i in range(num_levels):
            dec_num_patch = math.ceil(c_out / window_size[i])
            self.dec_num_patches.append(dec_num_patch)

        self.W_pos = nn.ParameterList()

        # for i in range(num_levels):
        #     self.W_pos.append(
        #         positional_encoding(
        #             pe,
        #             learn_pe,
        #             self.enc_num_patches[i] + self.dec_num_patches[i],
        #             _d_model,
        #         )
        #     )

        encoder = []
        projections = []
        for i in range(num_levels):
            encoder.append(
                MultiresPatchTSTEncoder(
                    c_in=c_in,
                    num_patch=self.enc_num_patches[i],
                    patch_len=window_size[i],
                    num_layers=enc_num_layers,
                    num_heads=enc_num_heads,
                    d_model=enc_d_model,
                    d_ff=enc_d_ff,
                    dropout=dropout,
                )
            )
            projections.append(
                nn.Linear(enc_d_model, 1 if i < num_levels - 1 else window_size[i])
            )

        self.encoder = nn.ModuleList(encoder)
        self.projections = nn.ModuleList(projections)

        decoder = []
        for i in range(num_levels):
            decoder.append(
                MultiresPatchTSTDecoder(
                    c_in=c_in,
                    num_patch=self.enc_num_patches[i] + self.dec_num_patches[i],
                    dec_num_patch=self.dec_num_patches[i],
                    patch_len=window_size[i],
                    num_layers=enc_num_layers,
                    num_heads=enc_num_heads,
                    d_model=enc_d_model,
                    d_ff=enc_d_ff,
                    dropout=dropout,
                )
            )

        self.decoder = nn.ModuleList(decoder)

    def forward(self, x, y, padding_mask=None, X_time=None, y_time=None):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        # prepare positional encodings

        # TODO best input would be [bs x seq_len x n_vars]

        bs, num_patch, n_vars, patch_len = x.shape

        # TODO first without revin, maybe later revin per level

        if self.revin:
            x = x.transpose(1, 2)
            x = x.reshape(bs, n_vars, num_patch * patch_len)  # bs x n_vars x seq_len
            x = x.transpose(1, 2)  # bs x seq_len x n_vars

            x = self.revin_layer(x, mode="norm")

            x = x.transpose(1, 2)
            x = x.reshape(bs, n_vars, num_patch, patch_len)
            x = x.transpose(1, 2)

        z_enc = []
        enc_layer_target = []
        enc_layer_pred = []

        for i, layer in enumerate(self.encoder):
            # patching
            x = create_patch(
                x.squeeze(), patch_len=layer.patch_len, stride=layer.patch_len
            )

            # x: [bs x num_patch x nvars x d_model]
            x = x.transpose(1, 2)
            # x: [bs x nvars x num_patch x d_model]
            x = x.reshape(bs * n_vars, layer.num_patch, layer.patch_len)
            # x: [bs * nvars x num_patch x d_model]

            z_l = layer(x)
            z_enc.append(z_l)

            z_l = self.projections[i](z_l)

            if i < self.num_levels - 1:
                z_l = z_l.squeeze(-1)

                if self.interpolation:  # linear interpolation
                    z_l = z_l.unsqueeze(1)
                    z_l = torch.nn.functional.interpolate(
                        z_l, size=layer.num_patch * layer.patch_len, mode="linear"
                    )
                    z_l = z_l.squeeze(1)
                else:  # step function
                    # TODO if we use step function, implement usage of window size correctly here, probably -i -1
                    z_l = z_l.unsqueeze(1)
                    z_l = torch.nn.functional.interpolate(
                        z_l, size=layer.num_patch * layer.patch_len, mode="nearest"
                    )
                    z_l = z_l.squeeze(1)
            else:
                z_l = z_l.reshape(bs * n_vars, layer.num_patch * layer.patch_len)

            z_l = z_l.reshape(bs, n_vars, layer.num_patch * layer.patch_len)
            x = x.reshape(bs, n_vars, layer.num_patch, layer.patch_len)
            x = x.reshape(bs, n_vars, -1)

            enc_layer_target.append(x)
            enc_layer_pred.append(z_l)

            x = x - z_l

            x = x.transpose(1, 2)

        z_dec = []
        dec_layer_pred = []
        dec_layer_target = []

        for i, layer in enumerate(self.decoder):
            z_p = layer(z_enc[i])
            z_dec.append(z_p)
            z_p = self.projections[i](z_p)

            if i < self.num_levels - 1:
                z_p = z_p.squeeze(-1)

                if self.interpolation:  # linear interpolation
                    z_p = z_p.unsqueeze(1)
                    z_p = torch.nn.functional.interpolate(
                        z_p, size=layer.dec_num_patch * layer.patch_len, mode="linear"
                    )
                    z_p = z_p.squeeze(1)
                else:  # step function
                    # TODO if we use step function, implement usage of window size correctly here, probably -i -1
                    z_p = z_p.unsqueeze(1)
                    z_p = torch.nn.functional.interpolate(
                        z_p, size=layer.dec_num_patch * layer.patch_len, mode="nearest"
                    )
                    z_p = z_p.squeeze(1)

            else:
                z_p = z_p.reshape(bs * n_vars, layer.dec_num_patch * layer.patch_len)

            z_p = z_p[:, : self.pred_len]
            z_p = z_p.reshape(bs, n_vars, self.pred_len).transpose(1, 2)
            dec_layer_target.append(y)
            dec_layer_pred.append(z_p)
            y = y - z_p

        if self.revin:
            z_final = self.revin_layer(z_final, mode="denorm")

        return enc_layer_target, enc_layer_pred, dec_layer_target, dec_layer_pred


class MultiresPatchTSTEncoder(nn.Module):
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
        res_attention=False,
        task=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.task = task
        self.patch_len = patch_len
        self.num_patch = num_patch

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
        bs_nvars, num_patch, patch_len = x.shape

        # input encoding
        # if not self.shared_embedding:
        #     x_out = []
        #     for i in range(n_vars):
        #         z = self.W_P[i](x[:, :, i, :])
        #         x_out.append(z)
        #     x = torch.stack(x_out, dim=2)
        # else:
        x = self.W_P(x)

        # # x: [bs x num_patch x nvars x d_model]
        # x = x.transpose(1, 2)
        # # x: [bs x nvars x num_patch x d_model]
        # x = x.reshape(bs * n_vars, num_patch, self.d_model)
        # # x: [bs * nvars x num_patch x d_model]

        # add positional encoding
        x = self.dropout(x + self.W_pos)

        # apply transformer encoder
        z = self.encoder(x)

        # z = z.reshape(bs, n_vars, num_patch, self.d_model)
        # # z: [bs x nvars x num_patch x d_model]

        # z = z.transpose(2, 3)
        # # z: [bs x nvars x d_model x num_patch]

        return z


class MultiresPatchTSTDecoder(nn.Module):
    def __init__(
        self,
        c_in,
        num_patch,
        dec_num_patch,
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
        res_attention=False,
        task=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.task = task
        self.dec_num_patch = dec_num_patch
        self.patch_len = patch_len

        # input encoding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(c_in):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        self.decoder = TSTDecoder(
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

    def forward(self, cross) -> Tensor:
        bs_nvars, num_patch, d_model = cross.shape

        x = torch.repeat_interleave(self.start_token, dim=1, repeats=self.dec_num_patch)
        x = torch.repeat_interleave(x, dim=0, repeats=bs_nvars)

        # add positional encoding
        cross = self.dropout(cross + self.W_pos[: cross.shape[1], :])
        x = self.dropout(x + self.W_pos[cross.shape[1] :, :])

        # apply transformer encoder
        z = self.decoder(x=x, cross=cross)

        # z = z.reshape(bs, n_vars, num_patch, self.d_model)
        # # z: [bs x nvars x num_patch x d_model]

        # z = z.transpose(2, 3)
        # # z: [bs x nvars x d_model x num_patch]

        return z
