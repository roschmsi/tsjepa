# References:
# https://github.com/facebookresearch/mae
# https://github.com/yuqinie98/PatchTST

import torch
import torch.nn as nn
from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.pos_encoding import positional_encoding
from models.patch_tst_t.model import ClassificationHead, PredictionHead


class MAEEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_patch: int,
        patch_len: int,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
        dropout: float,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token=False,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
        store_attn: bool = False,
    ):
        super().__init__()
        self.enc_d_model = enc_d_model

        # input encoding
        self.W_P = nn.Linear(patch_len * c_in, enc_d_model)

        # tokens
        self.cls_token = (
            nn.Parameter(torch.zeros(1, enc_d_model)) if cls_token else None
        )

        # encoder positional encoding
        self.encoder_pos_embed = positional_encoding(
            pe, learn_pe, num_patch, enc_d_model
        )

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = TSTEncoder(
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

    def forward(self, x, padding_mask=None, target_masks=None):
        bs, num_patch, n_vars, patch_len = x.shape

        x = torch.reshape(x, (bs, num_patch, n_vars * patch_len))

        # input encoding
        x = self.W_P(x)
        # x: [bs x num_patch x d_model]

        # add positional encoding
        encoder_pos_embed = self.encoder_pos_embed.expand(bs, -1, -1)
        if target_masks is not None:
            # inverse target mask
            target_masks = ~target_masks.bool()
            # target_masks: [bs x num_patch x n_vars], all n_vars are the same
            target_masks = target_masks[:, :, 0]
            # target_masks: [bs x num_patch]
            target_masks = target_masks.unsqueeze(-1).expand(-1, -1, self.enc_d_model)
            # target_masks: [bs x num_patch x d_model]
            encoder_pos_embed = encoder_pos_embed[target_masks.bool()].reshape(
                bs, num_patch, self.enc_d_model
            )

        x = self.dropout(x + encoder_pos_embed)

        # append class token
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # apply transformer encoder
        x = self.encoder(x)
        # x: [bs x num_patch x d_model]

        return x


class MAEDecoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_patch: int,
        patch_len: int,
        enc_d_model: int,
        dec_num_layers: int,
        dec_num_heads: int,
        dec_d_model: int,
        dec_d_ff: int,
        dropout: float,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token=False,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
        store_attn: bool = False,
    ):
        super().__init__()
        self.dec_d_model = dec_d_model

        self.cls_token = cls_token  # here a boolen, no tensor

        self.decoder_embed = nn.Linear(enc_d_model, dec_d_model, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))

        # decoder positional encoding
        self.decoder_pos_embed = positional_encoding(
            pe, learn_pe, num_patch, dec_d_model
        )

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # decoder
        self.decoder = TSTEncoder(
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

        self.decoder_pred = nn.Linear(dec_d_model, c_in * patch_len)

    def forward(self, lat, ids_restore, padding_mask):
        # embed latent tokens
        # lat: [bs x num_patch x d_model]
        x = self.decoder_embed(lat)
        # x: [bs x num_patch x d_model]

        # remove class tokens
        if self.cls_token:
            cls_token = x[:, :1, :]
            x = x[:, 1:, :]

        # append and unshuffle mask tokens
        bs, num_patch, ch = ids_restore.shape
        # ids_restore = ids_restore.transpose(1, 2).reshape(bs * ch, length)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )

        x = torch.cat([x, mask_tokens], dim=1)
        ids_restore = ids_restore[:, :, 0]
        # ids_restore: [bs * num_patch]
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # add positional encoding
        x = self.dropout(x + self.decoder_pos_embed)

        # reappend channel and class tokens
        if self.cls_token:
            x = torch.cat([cls_token, x], dim=1)

        # apply transformer decoder
        x = self.decoder(x)
        # x: [bs x num_patch x d_model]

        # predictor projection
        x = self.decoder_pred(x)
        # x: [bs x num_patch x patch_len * n_vars]

        # prepare output, remove class token
        if self.cls_token:
            x = x[:, 1:, :]

        x = x.reshape(bs, num_patch, ch, -1)
        # x: [bs x num_patch x n_vars x patch_len]

        return x


# encode time (1d data)
class MaskedAutoencoderT(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_patch: int,
        patch_len: int,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
        dec_num_layers: int,
        dec_num_heads: int,
        dec_d_model: int,
        dec_d_ff: int,
        dropout: float,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token=False,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
        store_attn: bool = False,
    ):
        super().__init__()

        self.encoder = MAEEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            enc_num_layers=enc_num_layers,
            enc_num_heads=enc_num_heads,
            enc_d_model=enc_d_model,
            enc_d_ff=enc_d_ff,
            dropout=dropout,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            pe=pe,
            learn_pe=learn_pe,
            cls_token=cls_token,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
            store_attn=store_attn,
        )

        self.decoder = MAEDecoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            enc_d_model=enc_d_model,
            dec_num_layers=dec_num_layers,
            dec_num_heads=dec_num_heads,
            dec_d_model=dec_d_model,
            dec_d_ff=dec_d_ff,
            dropout=dropout,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            pe=pe,
            learn_pe=learn_pe,
            cls_token=cls_token,
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


class MaskedAutoencoderTPredictor(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_patch: int,
        patch_len: int,
        enc_num_layers: int,
        enc_num_heads: int,
        enc_d_model: int,
        enc_d_ff: int,
        dropout: float,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token: bool = False,
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

        self.encoder = MAEEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            enc_num_layers=enc_num_layers,
            enc_num_heads=enc_num_heads,
            enc_d_model=enc_d_model,
            enc_d_ff=enc_d_ff,
            dropout=dropout,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            pe=pe,
            learn_pe=learn_pe,
            cls_token=cls_token,
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
                individual=individual,
                n_vars=c_in,
                d_model=enc_d_model,
                num_patch=num_patch,
                forecast_len=c_out,
                head_dropout=head_dropout,
            )
        else:
            raise ValueError(f"Task {task} not defined.")

    def forward(self, imgs, padding_mask_kept):
        latent = self.encoder(imgs, padding_mask=padding_mask_kept)
        # latent: [bs x num_patch x d_model]

        # remove cls token
        if self.task != "classification" and self.cls_token:
            latent = latent[:, 1:, :]

        pred = self.head(latent)

        return pred
