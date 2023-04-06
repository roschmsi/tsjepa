import torch
import torch.nn as nn
from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.heads import PredictionHead
from models.patch_tst_tc.model import ClassificationHead
from models.patch_tst_tc.positional_embedding import get_2d_sincos_pos_embed


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
        shared_embedding=True,
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
        self.shared_embedding = shared_embedding
        self.enc_d_model = enc_d_model

        # input encoding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(c_in):
                self.W_P.append(nn.Linear(patch_len, enc_d_model))
        else:
            self.W_P = nn.Linear(patch_len, enc_d_model)

        # channel and class tokens
        self.cls_token = (
            nn.Parameter(torch.zeros(1, enc_d_model)) if cls_token else None
        )

        # encoder positional encoding
        self.encoder_pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(embed_dim=enc_d_model, len=num_patch, ch=c_in)
            ).float(),
            requires_grad=learn_pe,
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

        x = torch.reshape(x, (bs, num_patch * n_vars, self.enc_d_model))
        # x: [bs x nvars * num_patch x d_model]

        # add positional encoding
        encoder_pos_embed = self.encoder_pos_embed.expand(bs, -1, -1)
        if target_masks is not None:
            # target_masks: [bs x num_patch x n_vars]
            target_masks = target_masks.reshape(bs, num_patch * n_vars)
            # target_masks: [bs x num_patch * n_vars]
            target_masks = target_masks.unsqueeze(-1).expand(-1, -1, self.enc_d_model)
            encoder_pos_embed = encoder_pos_embed[target_masks.bool()].reshape(
                bs, num_patch * n_vars, self.enc_d_model
            )

        x = self.dropout(x + encoder_pos_embed)

        x = x.reshape(bs, num_patch * n_vars, self.enc_d_model)

        # append cls token
        encoder_pos_embed = self.encoder_pos_embed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # apply transformer encoder
        x = self.encoder(x)
        # x: [bs x num_patch * n_vars x d_model]

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
        shared_embedding=True,
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
        self.decoder_pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(embed_dim=dec_d_model, len=num_patch, ch=c_in)
            ).float(),
            requires_grad=learn_pe,
        )
        # decoder_pos_embed: [num_patch * n_vars, d_model]

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

        self.decoder_pred = nn.Linear(dec_d_model, patch_len)

    def forward(self, lat, ids_restore, padding_mask):
        # embed latent tokens
        # lat: [bs x n_vars * num_patch x d_model]
        x = self.decoder_embed(lat)
        # x: [bs x n_vars * num_patch x d_model]

        # remove class tokens
        if self.cls_token:
            cls_token = x[:, :1, :]
            x = x[:, 1:, :]

        # append and unshuffle mask tokens
        bs, length, ch = ids_restore.shape
        ids_restore = ids_restore.reshape(bs, length * ch)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )

        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # add positional encoding
        x = self.dropout(x + self.decoder_pos_embed)

        # reappend class tokens
        if self.cls_token:
            x = torch.cat([cls_token, x], dim=1)

        # apply transformer decoder
        x = self.decoder(x)
        # x: [bs x n_vars * num_patch x d_model]

        # predictor projection
        x = self.decoder_pred(x)
        # x: [bs x n_vars * num_patch x patch_len]

        # prepare output, remove class tokens
        if self.cls_token:
            x = x[:, 1:, :]

        x = x.reshape(bs, ch, length, -1).transpose(1, 2)
        # x: [bs x num_patch x n_vars x patch_len]

        return x


# encode time and channel (2d data)
class MaskedAutoencoderTC(nn.Module):
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
        shared_embedding=True,
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
            shared_embedding=shared_embedding,
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
            shared_embedding=shared_embedding,
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


class MaskedAutoencoderTCPredictor(nn.Module):
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
        shared_embedding=True,
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
            shared_embedding=shared_embedding,
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

    def forward(self, imgs, padding_mask):
        bs, num_patch, n_vars, patch_len = imgs.shape

        latent = self.encoder(imgs, padding_mask=padding_mask)
        # latent: [bs x num_patch * n_vars x d_model]

        # remove cls token
        if self.task != "classification" and self.cls_token:
            latent = latent[:, 1:, :]
        if self.task == "forecasting":
            latent = latent.reshape(bs, num_patch, n_vars, -1)
            latent = latent.permute(0, 2, 3, 1)
            # latent: [bs, n_vars, d_model, num_patch]

        pred = self.head(latent)

        return pred
