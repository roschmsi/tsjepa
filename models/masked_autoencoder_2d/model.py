# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
from models.patch_tst.model import (
    ClassificationHead,
    TSTEncoder,
)
from models.patch_tst_2d.positional_embedding import get_2d_sincos_pos_embed


class MaskedAutoencoderTST2d(nn.Module):
    def __init__(
        self,
        num_patch: int,
        patch_len=16,
        masking_ratio=0.5,
        c_in=3,
        enc_d_model=256,
        enc_d_ff=512,
        enc_num_layers=8,
        enc_num_heads=8,
        dec_d_model=256,
        dec_d_ff=512,
        dec_num_layers=8,
        dec_num_heads=8,
        norm_layer=nn.LayerNorm,
        shared_embedding=True,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        cls_token=False,
        ch_token=False,
        task=None,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.n_vars = c_in
        self.enc_num_patch = int((1 - masking_ratio) * num_patch)
        self.patch_len = patch_len
        self.shared_embedding = shared_embedding
        self.enc_d_model = enc_d_model
        self.dec_d_model = dec_d_model
        self.task = task

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(patch_len, enc_d_model))
        else:
            self.W_P = nn.Linear(patch_len, enc_d_model)

        # Positional encoding
        self.encoder_pos_embed = (
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    embed_dim=enc_d_model, len=self.enc_num_patch, ch=c_in
                )
            )
            .float()
            .cuda()
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, enc_d_model)) if cls_token else None
        )

        self.ch_token = (
            nn.Parameter(torch.zeros(1, c_in, enc_d_model)) if ch_token else None
        )

        self.encoder = TSTEncoder(
            enc_d_model,
            enc_num_heads,
            d_ff=enc_d_ff,
            num_layers=enc_num_layers,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            store_attn=store_attn,
        )
        self.encoder_norm = norm_layer(enc_d_model)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(enc_d_model, dec_d_model, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))

        self.decoder_pos_embed = (
            torch.from_numpy(
                get_2d_sincos_pos_embed(embed_dim=dec_d_model, len=num_patch, ch=c_in)
            )
            .float()
            .cuda()
        )

        self.decoder = TSTEncoder(
            dec_d_model,
            dec_num_heads,
            d_ff=dec_d_ff,
            num_layers=dec_num_layers,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            store_attn=store_attn,
        )

        self.decoder_norm = norm_layer(dec_d_model)
        self.decoder_pred = nn.Linear(dec_d_model, patch_len)
        # --------------------------------------------------------------------------

        self.dropout = nn.Dropout(dropout)

    def forward_encoder(self, x, padding_mask=None):
        # embed patches
        bs, num_patch, n_vars, patch_len = x.shape

        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)  # x: [bs x num_patch x nvars x d_model]

        x = x.transpose(1, 2)  # x: [bs x nvars x num_patch x d_model]
        x = torch.reshape(
            x, (bs, n_vars * num_patch, self.enc_d_model)
        )  # u: [bs x nvars * num_patch x d_model]

        encoder_pos_embed = self.encoder_pos_embed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            num_patch += 1

            encoder_pos_embed = torch.cat(
                [torch.zeros([1, self.enc_d_model]).cuda(), encoder_pos_embed],
                axis=0,
            )

        x = self.dropout(x + encoder_pos_embed)  # u: [bs * nvars x num_patch x d_model]

        # apply Transformer blocks
        x = self.encoder(x, padding_mask)

        return x

    def forward_decoder(self, x, ids_restore, key_padding_mask):
        # embed tokens (x is encoder latent representation)
        # for debugging
        x = self.decoder_embed(x)

        bs, length, ch = ids_restore.shape
        # ids_restore = ids_restore.transpose(1, 2).reshape(bs * ch, length)
        ids_restore = ids_restore.reshape(bs, ch * length)

        # append mask tokens to sequence (without class token and ch token)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ch * ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        # reappend cls token
        if self.cls_token is not None:
            x = torch.cat([x[:, :1, :], x_], dim=1)
        else:
            x = x_

        # add pos embed
        decoder_pos_embed = self.decoder_pos_embed
        if self.cls_token is not None:
            decoder_pos_embed = torch.cat(
                [torch.zeros([1, self.dec_d_model]), decoder_pos_embed],
                axis=0,
            )
            length += 1

        x = x + decoder_pos_embed

        # apply Transformer blocks
        # x = x.squeeze()
        x = self.decoder(x, key_padding_mask=key_padding_mask)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        x = x.reshape(bs, ch, length, -1).transpose(1, 2)

        if self.task != "classification" and self.cls_token is not None:
            x = x[:, 1:, :, :]

        return x

    def forward(self, imgs, padding_mask, padding_mask_kept, ids_restore):
        latent = self.forward_encoder(imgs, padding_mask_kept)
        pred = self.forward_decoder(
            latent, ids_restore, key_padding_mask=padding_mask
        )  # [N, L, p*p*3]
        return pred


class MaskedAutoencoderTST2dClassifier(nn.Module):
    def __init__(
        self,
        num_patch: int,
        patch_len=16,
        masking_ratio=0.5,
        c_in=3,
        target_dim=27,
        enc_d_model=256,
        enc_d_ff=512,
        enc_num_layers=8,
        enc_num_heads=8,
        norm_layer=nn.LayerNorm,
        shared_embedding=True,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        head_dropout=0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.n_vars = c_in
        self.enc_num_patch = int((1 - masking_ratio) * num_patch)
        self.patch_len = patch_len
        self.shared_embedding = shared_embedding
        self.enc_d_model = enc_d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(patch_len, enc_d_model))
        else:
            self.W_P = nn.Linear(patch_len, enc_d_model)

        # Positional encoding
        self.encoder_pos_embed = (
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    embed_dim=enc_d_model, len=self.enc_num_patch, ch=c_in
                )
            )
            .float()
            .cuda()
        )

        self.encoder = TSTEncoder(
            enc_d_model,
            enc_num_heads,
            d_ff=enc_d_ff,
            num_layers=enc_num_layers,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            store_attn=store_attn,
        )
        self.encoder_norm = norm_layer(enc_d_model)

        self.dropout = nn.Dropout(dropout)

        self.head = ClassificationHead(
            n_vars=self.n_vars,
            d_model=enc_d_model,
            n_classes=target_dim,
            head_dropout=head_dropout,
        )

    def forward_encoder(self, x, padding_masks_kept=None):
        # embed patches
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)  # x: [bs x num_patch x nvars x d_model]

        x = x.transpose(1, 2)  # x: [bs x nvars x num_patch x d_model]
        x = torch.reshape(
            x, (bs, n_vars * num_patch, self.enc_d_model)
        )  # u: [bs x nvars * num_patch x d_model]

        encoder_pos_embed = self.encoder_pos_embed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            num_patch += 1

            encoder_pos_embed = torch.cat(
                [torch.zeros([1, self.enc_d_model]).cuda(), encoder_pos_embed],
                axis=0,
            )

        x = self.dropout(x + encoder_pos_embed)  # u: [bs x nvars * num_patch x d_model]

        # apply Transformer blocks
        z = self.encoder(x)

        return z

    def forward(self, imgs, padding_mask_kept):
        latent = self.forward_encoder(imgs, padding_mask_kept)
        # bs x nvars x d_model x num_patch
        latent = torch.reshape(
            latent, (imgs.shape[0], imgs.shape[2], imgs.shape[1], -1)
        ).transpose(2, 3)
        pred = self.head(latent)
        return pred
