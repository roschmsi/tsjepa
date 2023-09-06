# Reference: https://github.com/yuqinie98/PatchTST

import math
import torch
from torch import Tensor, nn
from models.hierarchical_patch_tst.mlp import DownsamplingMLP, UpsamplingMLP

from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.pos_encoding import positional_encoding
from models.hierarchical_patch_tst.decoder import TSTDecoder
from models.patch_tst.layers.revin import RevIN


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
        ch_factor,
        num_patch: int,
        patch_len: int,
        num_levels: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        window_size: int,
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
    ):
        super().__init__()
        self.use_time_features = use_time_features
        self.c_in = c_in
        self.layer_wise_prediction = layer_wise_prediction
        self.revin = revin

        if self.revin:
            self.revin_layer = RevIN(
                num_features=c_in, affine=True, subtract_last=False
            )

        enc_num_patch = num_patch
        self.enc_num_patches = [enc_num_patch]
        for i in range(num_levels - 1):
            enc_num_patch = math.ceil(enc_num_patch / window_size)
            self.enc_num_patches.append(enc_num_patch)

        dec_num_patch = math.ceil(c_out / patch_len)
        self.dec_num_patches = [dec_num_patch]
        for i in range(num_levels - 1):
            dec_num_patch = math.ceil(dec_num_patch / window_size)
            self.dec_num_patches.append(dec_num_patch)

        self.enc_max_patches = []
        self.dec_max_patches = []
        for i in range(num_levels):
            enc_max_patch = self.enc_num_patches[-1] * (window_size**i)
            self.enc_max_patches.append(enc_max_patch)

            dec_max_patch = self.dec_num_patches[-1] * (window_size**i)
            self.dec_max_patches.append(dec_max_patch)

        self.enc_max_patches = self.enc_max_patches[::-1]
        self.dec_max_patches = self.dec_max_patches[::-1]

        num_patch_enc_dec = (self.enc_num_patches[-1] + self.dec_num_patches[-1]) * (
            (window_size) ** (num_levels - 1)
        )

        self.W_pos = positional_encoding(pe, learn_pe, num_patch_enc_dec, d_model)

        pe_conv_layers = []
        in_channels = d_model
        out_channels = int(ch_factor * d_model)
        for i in range(num_levels - 1):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=window_size,
                stride=window_size,
            )
            pe_conv_layers.append(conv)

            in_channels = int(ch_factor * in_channels)
            out_channels = int(ch_factor * out_channels)
        self.pe_conv_layers = nn.ModuleList(pe_conv_layers)

        self.backbone = HierarchicalPatchTSTEncoder(
            c_in=c_in,
            ch_factor=ch_factor,
            patch_len=patch_len,
            num_levels=num_levels,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            window_size=window_size,
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
            pass  # TODO
        elif task == "forecasting":
            self.head = HierarchicalPatchTSTDecoder(
                c_in=c_in,
                ch_factor=ch_factor,
                dec_num_patches=self.dec_num_patches[::-1],
                patch_len=patch_len,
                num_levels=num_levels,
                num_layers=num_layers,
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                window_size=window_size,
                dropout=dropout,
                pe="sincos",
                norm=norm,
                pred_len=c_out,
                layer_wise_prediction=layer_wise_prediction,
            )

        elif task == "classification":
            pass  # TODO
        else:
            raise ValueError(f"Task {task} not defined.")

    def forward(self, z, padding_mask=None, X_time=None, y_time=None):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        # prepare positional encodings

        bs, num_patch, n_vars, patch_len = z.shape

        pos_encodings = []
        pe = self.W_pos.transpose(0, 1)
        pe_enc = pe[:, : self.enc_max_patches[0]]
        pe_dec = pe[:, self.enc_max_patches[0] :]
        pe_enc = pe_enc[:, -self.enc_num_patches[0] :]
        pe_dec = pe_dec[:, : self.dec_num_patches[0]]
        pe_enc_dec = torch.cat([pe_enc, pe_dec], dim=1)
        pos_encodings.append(pe_enc_dec.transpose(0, 1))

        # pos_encodings = [self.W_pos]

        pe = self.W_pos.transpose(0, 1)
        for i, conv in enumerate(self.pe_conv_layers):
            # pe = conv(pe)
            # pos_encodings.append(pe.transpose(0, 1))
            pe = conv(pe)
            pe_enc = pe[:, : self.enc_max_patches[i + 1]]
            pe_dec = pe[:, self.enc_max_patches[i + 1] :]
            pe_enc = pe_enc[:, -self.enc_num_patches[i + 1] :]
            pe_dec = pe_dec[:, : self.dec_num_patches[i + 1]]
            pe_enc_dec = torch.cat([pe_enc, pe_dec], dim=1)
            pos_encodings.append(pe_enc_dec.transpose(0, 1))

        if self.revin:
            z = z.transpose(1, 2)
            z = z.reshape(bs, n_vars, num_patch * patch_len)  # bs x n_vars x seq_len
            z = z.transpose(1, 2)  # bs x seq_len x n_vars

            z = self.revin_layer(z, mode="norm")

            z = z.transpose(1, 2)
            z = z.reshape(bs, n_vars, num_patch, patch_len)
            z = z.transpose(1, 2)

        time_encodings = None
        z = self.backbone(z, pe=pos_encodings, te=time_encodings)
        # z: [bs x nvars x d_model x num_patch]

        z = self.head(
            z[::-1],
            pe=pos_encodings[::-1],
            te=time_encodings[::-1] if time_encodings is not None else None,
        )

        if self.layer_wise_prediction:
            z_final = z[0]

            # TODO fix if there is no layer wise prediction

            if self.revin:
                z_final = self.revin_layer(z_final, mode="denorm")

            #     z_dec = []

            #     for z_layer in z[1]:
            #         z_layer = self.revin_layer(z_layer, mode="denorm")
            #         z_dec.append(z_layer)

            # else:
            z_dec = z[1]

            return z_final, z_dec

        else:
            z_final = z

            if self.revin:
                z_final = self.revin_layer(z, mode="denorm")

            return z_final


class HierarchicalPatchTSTEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        ch_factor,
        patch_len,
        num_levels,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        window_size,
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

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        enc_modules = []

        d_layer = d_model

        for i in range(num_levels - 1):
            # encoder
            encoder = TSTEncoder(
                num_layers=num_layers,
                num_heads=num_heads,
                d_model=d_layer,
                d_ff=2 * d_layer,
                dropout=dropout,
                norm=norm,
                pre_norm=pre_norm,
                activation=activation,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
                store_attn=store_attn,
            )
            mlp = DownsamplingMLP(
                c_in=d_layer,
                c_out=int(d_layer * ch_factor),
                win_size=window_size,
            )
            enc_modules.append(nn.Sequential(encoder, mlp))

            d_layer = int(d_layer * ch_factor)

        enc_modules.append(
            nn.Sequential(
                TSTEncoder(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    d_model=d_layer,
                    d_ff=2 * d_layer,
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

    def forward(self, x, pe=None, te=None) -> Tensor:
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
        x = self.dropout(x + pe[0][: x.shape[1]])
        # x = x.reshape(bs, n_vars, num_patch, self.d_model)

        # apply transformer encoder
        x_enc = []

        for i, (name, module) in enumerate(self.encoder.named_children()):
            for name, m_layer in module.named_children():
                if type(m_layer) is TSTEncoder:
                    # add positional encoding
                    # x = x + pe[i][: x.shape[1], :]
                    # x = torch.cat([x, te[i][:, : x.shape[1], :]], dim=2)
                    pass

                x = m_layer(x)

                if type(m_layer) is TSTEncoder:
                    _, num_patch, d_layer = x.shape
                    x_enc.append(x.reshape(bs, n_vars, num_patch, d_layer))

        return x_enc


class HierarchicalPatchTSTDecoder(nn.Module):
    def __init__(
        self,
        c_in,
        ch_factor,
        dec_num_patches,
        patch_len,
        num_levels,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        window_size,
        dropout,
        pred_len,
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
        layer_wise_prediction=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.task = task
        self.dec_num_patches = dec_num_patches
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.window_size = window_size
        self.num_levels = num_levels

        self.layer_wise_prediction = layer_wise_prediction

        # input encoding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(c_in):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        dec_modules = []

        d_layer = int(d_model * (ch_factor ** (num_levels - 1)))
        # p_layer = patch_len * (window_size ** (num_levels - 1))

        projections = []

        # TODO learn start token for full sequence
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_layer))

        for i in range(num_levels - 1):
            # encoder
            decoder = TSTDecoder(
                num_layers=num_layers,
                num_heads=num_heads,
                d_model=d_layer,
                d_ff=2 * d_layer,
                dropout=dropout,
                norm=norm,
                pre_norm=pre_norm,
                activation=activation,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
                store_attn=store_attn,
                num_patch=dec_num_patches[i],
            )
            mlp = UpsamplingMLP(
                c_in=d_layer,
                c_out=int(d_layer // ch_factor),
                win_size=window_size,  # , norm_layer=nn.BatchNorm1d
            )
            dec_modules.append(nn.Sequential(decoder, mlp))

            projections.append(nn.Linear(d_layer, 1))

            d_layer = int(d_layer // ch_factor)
            # p_layer = p_layer // window_size

        dec_modules.append(
            nn.Sequential(
                TSTDecoder(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    d_model=d_layer,
                    d_ff=2 * d_layer,
                    dropout=dropout,
                    norm=norm,
                    pre_norm=pre_norm,
                    activation=activation,
                    attn_dropout=attn_dropout,
                    res_attention=res_attention,
                    store_attn=store_attn,
                    num_patch=dec_num_patches[-1],
                )
            )
        )

        self.decoder = nn.Sequential(*dec_modules)
        projections.append(nn.Linear(d_layer, patch_len))

        self.projections = nn.ModuleList(projections)
        # self.projection = nn.Linear(d_layer, patch_len)

    def forward(self, cross, pe=None, te=None):
        bs, n_vars, num_patch, d_model = cross[0].shape

        x = torch.repeat_interleave(
            self.start_token, dim=1, repeats=self.dec_num_patches[0]
        )
        x = torch.repeat_interleave(x, dim=0, repeats=bs * n_vars)

        x_dec = []

        for i, (name, module) in enumerate(self.decoder.named_children()):
            for name, m_layer in module.named_children():
                # x = self.dropout(x + self.W_pos[i].cuda())
                if type(m_layer) is TSTDecoder:
                    bs, n_vars, num_patch_layer, d_layer = cross[i].shape
                    cross_i = cross[i].reshape(bs * n_vars, num_patch_layer, d_layer)

                    # positional encoding
                    cross_i = cross_i + pe[i][: cross_i.shape[1], :]
                    x = x + pe[i][cross_i.shape[1] :, :]

                    # time encoding
                    if te is not None:
                        cross_i = torch.cat(
                            [cross_i, te[i][:, : cross_i.shape[1], :]], dim=2
                        )
                        x = torch.cat([x, te[i][:, cross_i.shape[1] :, :]], dim=2)

                    x = m_layer(x=x, cross=cross_i)

                    x_dec.append(x)

                if type(m_layer) is UpsamplingMLP:
                    x = m_layer(x)

                    # TODO shorten upsampled sequences
                    if i < len(self.dec_num_patches) - 1:
                        x = x[:, : self.dec_num_patches[i + 1], :]

        if self.layer_wise_prediction:
            lbl_pred = []

            p_layer = self.patch_len * (self.window_size ** (self.num_levels - 1))

            for i, x_layer in enumerate(x_dec):
                x_proj = self.projections[i](x_layer)
                if i < len(x_dec) - 1:
                    x_proj = torch.repeat_interleave(x_proj, dim=2, repeats=p_layer)
                x_proj = x_proj.reshape(bs * n_vars, -1)
                x_proj = x_proj.reshape(bs, n_vars, -1).transpose(1, 2)
                x_proj = x_proj[:, : self.pred_len, :]
                lbl_pred.append(x_proj)

                p_layer = p_layer // self.window_size

            x = torch.stack(lbl_pred, dim=0).sum(dim=0)

            return x, lbl_pred

        else:
            x = self.projections[-1](x)
            x = x.reshape(bs * n_vars, -1)
            x = x.reshape(bs, n_vars, -1).transpose(1, 2)
            x = x[:, : self.pred_len, :]

            return x
