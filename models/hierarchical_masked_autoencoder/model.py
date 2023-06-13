# References:
# https://github.com/facebookresearch/mae
# https://github.com/yuqinie98/PatchTST

import torch
import torch.nn as nn
from models.hierarchical_patch_tst.model import DownsamplingMLP, UpsamplingMLP
from models.patch_tst.layers.pos_encoding import positional_encoding
from models.patch_tst.model import (
    ClassificationHead,
    TSTEncoder,
)


class PredictionHead(nn.Module):
    def __init__(self, n_vars, d_model, num_patch, forecast_len, head_dropout):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(32 * 128, forecast_len),
                nn.Linear(64 * 128, forecast_len),
                nn.Linear(128 * 128, forecast_len),
            ]
        )

        self.mix = nn.Linear(3 * forecast_len, forecast_len)

    def forward(self, x_enc):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        # bs, num_patch, d_model = x.shape

        forecasted = []

        for i in range(len(x_enc)):
            x = self.flatten(x_enc[-(i + 1)])
            x = self.mlp[i](x)
            x = self.dropout(x)
            forecasted.append(x)

        y = torch.cat(forecasted, dim=-1)
        y = self.mix(y)

        y = y.transpose(1, 2)

        return y


class HMAEEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_patch: int,
        patch_len: int,
        num_levels,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
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
    ):
        super().__init__()
        self.shared_embedding = shared_embedding
        self.enc_d_model = enc_d_model
        self.num_levels = num_levels

        # input encoding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(c_in):
                self.W_P.append(nn.Linear(4, enc_d_model))
        else:
            self.W_P = nn.Linear(4, enc_d_model)

        # channel and class tokens
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, enc_d_model)) if cls_token else None
        )
        self.ch_token = (
            nn.Parameter(torch.zeros(c_in, 1, enc_d_model)) if ch_token else None
        )

        # encoder positional encoding
        # TODO positional encoding on every level ?
        self.encoder_pos_embed = positional_encoding(
            pe, learn_pe, num_patch, enc_d_model
        )

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        modules = []

        for i in range(num_levels - 1):
            # encoder
            encoder = TSTEncoder(
                num_layers=enc_num_layers,
                num_heads=enc_num_heads,
                d_model=enc_d_model,
                d_ff=enc_d_ff,
                dropout=dropout,
                norm=norm,
                pre_norm=pre_norm,
                activation=activation,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
                store_attn=store_attn,
            )
            mlp = DownsamplingMLP(enc_d_model, win_size=2)
            modules.append(nn.Sequential(encoder, mlp))

        modules.append(
            nn.Sequential(
                TSTEncoder(
                    num_layers=enc_num_layers,
                    num_heads=enc_num_heads,
                    d_model=enc_d_model,
                    d_ff=enc_d_ff,
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

        self.encoder = nn.Sequential(*modules)

    def forward(self, x, padding_mask=None, target_masks=None):
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
        x = x.reshape(bs * n_vars, num_patch, self.enc_d_model)
        # x: [bs * nvars x num_patch x d_model]

        # add positional encoding
        encoder_pos_embed = self.encoder_pos_embed.expand(bs * n_vars, -1, -1)
        if target_masks is not None:
            # inverse target mask: 0 is keep, 1 is remove --> thus inverse
            target_masks = ~target_masks.bool()
            # target_masks: [bs x num_patch x n_vars]
            target_masks = target_masks.transpose(1, 2)
            target_masks = target_masks.reshape(bs * n_vars, -1)
            # target_masks: [bs * n_vars x num_patch]
            target_masks = target_masks.unsqueeze(-1).expand(-1, -1, self.enc_d_model)
            # target_masks: [bs * n_vars x num_patch x d_model]
            encoder_pos_embed = encoder_pos_embed[target_masks.bool()].reshape(
                bs * n_vars, -1, self.enc_d_model
            )

        x = self.dropout(x + encoder_pos_embed)

        x = x.reshape(bs, n_vars, num_patch, self.enc_d_model)

        # append channel and class token
        if self.ch_token is not None:
            ch_token = self.ch_token.expand(bs, -1, -1, -1)
            x = torch.cat((x, ch_token), dim=2)
            num_patch += 1
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(bs, n_vars, -1, -1)
            x = torch.cat((cls_token, x), dim=2)
            num_patch += 1

        x = x.reshape(bs * n_vars, num_patch, self.enc_d_model)

        x_enc = []

        for i, (name, module) in enumerate(self.encoder.named_children()):
            for name, m_layer in module.named_children():
                # pos encoding
                # x = self.dropout(x + self.W_pos[i].cuda())
                x = m_layer(x)

                if type(m_layer) is TSTEncoder:
                    x_enc.append(x.reshape(bs, n_vars, -1, self.enc_d_model))

        return x_enc


class HMAEDecoder(nn.Module):
    def __init__(
        self,
        num_patch: int,
        patch_len: int,
        enc_d_model: int,
        num_levels: int,
        dec_num_layers: int,
        dec_num_heads: int,
        dec_d_model: int,
        dec_d_ff: int,
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
    ):
        super().__init__()
        self.shared_embedding = shared_embedding
        self.dec_d_model = dec_d_model

        self.cls_token = cls_token  # here a boolen, no tensor
        self.ch_token = ch_token  # here a boolean, no tensor

        self.decoder_embed = nn.Linear(enc_d_model, dec_d_model, bias=True)

        self.mask_token = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, dec_d_model)),
                nn.Parameter(torch.zeros(1, 1, dec_d_model)),
                nn.Parameter(torch.zeros(1, 1, dec_d_model)),
                nn.Parameter(torch.zeros(1, 1, dec_d_model)),
            ]
        )

        # decoder positional encoding
        self.decoder_pos_embed = positional_encoding(
            pe, learn_pe, num_patch // (2 ** (num_levels - 1)), dec_d_model
        )

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # decoder
        dec_modules = []

        # encoder
        encoder = TSTEncoder(
            num_layers=dec_num_layers,
            num_heads=dec_num_heads,
            d_model=dec_d_model,
            d_ff=dec_d_ff,
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
                        d_model=dec_d_model, win_size=2, norm_layer=nn.BatchNorm1d
                    ),
                    nn.Linear(2 * dec_d_model, dec_d_model),
                    TSTEncoder(
                        num_layers=dec_num_layers,
                        num_heads=dec_num_heads,
                        d_model=dec_d_model,
                        d_ff=dec_d_ff,
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

        self.final_proj = nn.Linear(dec_d_model, patch_len)

    def forward(self, x_enc, ids_restore, padding_mask=None):
        # embed latent tokens
        y = self.decoder_embed(x_enc[-1])
        # x: [bs * n_vars x num_patch x d_model]
        bs, n_vars, num_patch, patch_len = y.shape
        y = y.reshape(bs * n_vars, -1, self.dec_d_model)

        # remove channel and class tokens
        if self.ch_token:
            ch_token = y[:, -1:, :]
            y = y[:, :-1, :]
        if self.cls_token:
            cls_token = y[:, :1, :]
            y = y[:, 1:, :]

        # append and unshuffle mask tokens
        bs, num_patch, ch = ids_restore.shape
        # ids_restore: [bs x num_patch x n_vars]
        ids_restore = ids_restore.transpose(1, 2).reshape(bs * ch, num_patch)
        # ids_restore: [bs * n_vars x num_patch]
        mask_tokens = self.mask_token[0].repeat(
            y.shape[0], ids_restore.shape[1] - y.shape[1], 1
        )

        y = torch.cat([y, mask_tokens], dim=1)
        y = torch.gather(
            y, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, y.shape[2])
        )

        # add positional encoding
        y = self.dropout(y + self.decoder_pos_embed)

        # reappend channel and class tokens
        if self.ch_token:
            y = torch.cat([y, ch_token], dim=1)
        if self.cls_token:
            y = torch.cat([cls_token, y], dim=1)

        # apply hierarchical decoder
        y_dec = []

        for i, (name, module) in enumerate(self.decoder.named_children()):
            for name, m_layer in module.named_children():
                # concatenate encoder outputs
                if type(m_layer) is nn.Linear:
                    x = x_enc[-i - 1]
                    x = x.reshape(bs * n_vars, -1, self.dec_d_model)
                    # unpool with mask tokens

                    # rearrange ids for multi resolution
                    ids_restore_exp = ids_restore.expand(2**i, -1, -1).permute(
                        1, 2, 0
                    )
                    ids_restore_x = torch.zeros_like(ids_restore_exp)
                    for j in range(ids_restore_exp.shape[2]):
                        ids_restore_x[:, :, j] = ids_restore_exp[:, :, j] * 2 + j
                    ids_restore_x = ids_restore_x.reshape(bs * ch, -1)

                    # ids_restore: [bs * n_vars x num_patch]
                    mask_tokens = self.mask_token[i].repeat(
                        x.shape[0], ids_restore_x.shape[1] - x.shape[1], 1
                    )
                    x = torch.cat([x, mask_tokens], dim=1)
                    x = torch.gather(
                        x,
                        dim=1,
                        index=ids_restore_x.unsqueeze(-1).repeat(1, 1, x.shape[2]),
                    )

                    # TODO ids_restore on higher levels
                    y = torch.cat([y, x], dim=2)

                y = m_layer(y)

                if type(m_layer) is TSTEncoder:
                    y_dec.append(
                        y.reshape(bs, n_vars, -1, self.dec_d_model).transpose(1, 2)
                    )

        # predictor projection
        pred = self.final_proj(y_dec[-1])
        # x: [bs * n_vars x num_patch x patch_len]

        # prepare output, remove class and channel token
        if self.ch_token:
            pred = pred[:, :-1, :]
        if self.cls_token:
            pred = pred[:, 1:, :]

        # y = y.reshape(bs, ch, num_patch, -1).transpose(1, 2)
        # x: [bs x num_patch x n_vars x patch_len]

        return pred


class HierarchicalMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_patch: int,
        patch_len: int,
        num_levels: int,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
        dec_num_layers: int,
        dec_num_heads: int,
        dec_d_model: int,
        dec_d_ff: int,
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
    ):
        super().__init__()

        self.encoder = HMAEEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            num_levels=num_levels,
            enc_num_layers=enc_num_layers,
            enc_num_heads=enc_num_heads,
            enc_d_model=enc_d_model,
            enc_d_ff=enc_d_ff,
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
        )

        self.decoder = HMAEDecoder(
            num_patch=num_patch,
            patch_len=patch_len,
            enc_d_model=enc_d_model,
            num_levels=num_levels,
            dec_num_layers=dec_num_layers,
            dec_num_heads=dec_num_heads,
            dec_d_model=dec_d_model,
            dec_d_ff=dec_d_ff,
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
        )

    def forward(self, imgs, padding_mask, padding_mask_kept, ids_restore, target_masks):
        latent = self.encoder(
            imgs, padding_mask=padding_mask_kept, target_masks=target_masks
        )
        pred = self.decoder(latent, ids_restore, padding_mask=padding_mask)
        return pred


class HierarchicalMaskedAutoencoderPredictor(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_patch: int,
        patch_len: int,
        num_levels: int,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
        dropout: float,
        shared_embedding=True,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token: bool = False,
        ch_token=False,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
        store_attn: bool = False,
        task=None,
        head_dropout=0,
        individual=False,
    ):
        super().__init__()
        self.task = task
        self.cls_token = cls_token
        self.ch_token = ch_token
        self.enc_d_model = enc_d_model

        self.encoder = HMAEEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            num_levels=num_levels,
            enc_num_layers=enc_num_layers,
            enc_num_heads=enc_num_heads,
            enc_d_model=enc_d_model,
            enc_d_ff=enc_d_ff,
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
        )

        if task == "classification":
            self.head = ClassificationHead(
                n_vars=c_in,
                d_model=enc_d_model,
                n_classes=c_out,
                head_dropout=head_dropout,
            )
        elif task == "forecasting":
            self.head = PredictionHead(
                n_vars=c_in,
                d_model=enc_d_model,
                num_patch=num_patch,
                forecast_len=c_out,
                head_dropout=head_dropout,
            )
        else:
            raise ValueError(f"Task {task} not defined.")

    def forward(self, imgs, padding_mask):
        bs, num_patch, n_vars, patch_len = imgs.shape

        latent = self.encoder(imgs, padding_mask=padding_mask)

        latent_reshaped = []

        for l in latent:
            latent_reshaped.append(l.reshape(bs, n_vars, -1, self.enc_d_model))

        pred = self.head(latent_reshaped)

        # # latent: [bs * nvars x num_patch x d_model]
        # latent = latent.reshape(bs, n_vars, -1, self.enc_d_model)
        # # latent: [bs x nvars x num_patch x d_model]

        # # remove ch token and cls token
        # if self.ch_token:
        #     latent = latent[:, :, :-1, :]
        # if self.task != "classification" and self.cls_token:
        #     latent = latent[:, :, 1:, :]

        # latent = latent.transpose(2, 3)
        # # latent: [bs x nvars x d_model x num_patch]

        # pred = self.head(latent)

        return pred
