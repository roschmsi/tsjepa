import torch
import torch.nn as nn
import math
from models.ts_jepa.model import get_1d_sincos_pos_embed
from models.ts_jepa.mask import apply_masks
from models.ts_jepa.tensors import trunc_normal_
from models.ts2vec.encoder import TransformerEncoderLayer  # , BatchNorm, LayerNorm


def get_predictor(config, max_seq_len):
    if config.predictor == "linear" and config.bert:
        predictor = nn.Linear(config.enc_d_model, config.patch_len)
    elif config.predictor == "transformer" and config.bert:
        predictor = TransformerPredictor(
            num_patches=int(max_seq_len // config.patch_len),
            encoder_embed_dim=config.enc_d_model,
            predictor_embed_dim=config.dec_d_model,
            depth=config.dec_num_layers,
            num_heads=config.dec_num_heads,
            mlp_ratio=config.dec_mlp_ratio,
            drop_rate=config.dropout,
            attn_drop_rate=config.attn_drop_rate,
            activation=config.activation,
            activation_drop_rate=config.activation_drop_rate,
            norm=config.norm,
            layer_norm_first=config.layer_norm_first,
            learn_pe=config.learn_pe,
            target_dim=config.patch_len,
        )
    elif config.predictor == "linear":
        predictor = nn.Linear(config.enc_d_model, config.enc_d_model)
    elif config.predictor == "mlp":
        predictor = nn.Sequential(
            nn.Linear(config.enc_d_model, config.dec_d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dec_d_model, config.enc_d_model),
        )
    elif config.predictor == "cnn":
        predictor = CNNPredictor(
            num_patches=int(max_seq_len // config.patch_len),
            encoder_embed_dim=config.enc_d_model,
            predictor_embed_dim=config.dec_d_model,
            depth=config.dec_num_layers,
            kernel_size=config.kernel_size,
            mask_noise_std=config.mask_noise_std,
            # decoder_groups=config.dec_groups,
            # projection_layers=config.dec_num_layers,
            # projection_ratio=config.dec_mlp_ratio,
        )
    elif config.predictor == "transformer":
        predictor = TransformerPredictor(
            num_patches=int(max_seq_len // config.patch_len),
            encoder_embed_dim=config.enc_d_model,
            predictor_embed_dim=config.dec_d_model,
            depth=config.dec_num_layers,
            num_heads=config.dec_num_heads,
            mlp_ratio=config.dec_mlp_ratio,
            drop_rate=config.dropout,
            attn_drop_rate=config.attn_drop_rate,
            activation=config.activation,
            activation_drop_rate=config.activation_drop_rate,
            norm=config.norm,
            layer_norm_first=config.layer_norm_first,
            learn_pe=config.learn_pe,
            target_dim=config.enc_d_model,
        )
    else:
        raise NotImplementedError

    return predictor


class TransformerPredictor(nn.Module):
    """Time Series Transformer with channel independence"""

    def __init__(
        self,
        num_patches,
        encoder_embed_dim,
        predictor_embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        target_dim,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        activation="gelu",
        activation_drop_rate=0.0,
        norm="LayerNorm",
        init_std=0.02,
        layer_norm_first=True,
        learn_pe=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = predictor_embed_dim
        self.layer_norm_first = layer_norm_first

        self.predictor_embed = nn.Linear(
            encoder_embed_dim, predictor_embed_dim, bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # 1d pos embed
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=learn_pe
        )
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1], num_patches, cls_token=False
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )

        self.predictor_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim=predictor_embed_dim,
                    ffn_embedding_dim=int(predictor_embed_dim * mlp_ratio),
                    num_attention_heads=num_heads,
                    dropout=drop_rate,
                    attention_dropout=attn_drop_rate,
                    activation_dropout=activation_drop_rate,
                    activation_fn=activation,
                    # norm_layer=norm_layer,
                    layer_norm_first=layer_norm_first,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)

        self.predictor_proj = nn.Linear(predictor_embed_dim, target_dim)

    def forward(self, x, ids_restore):
        # x: [bs x enc_num_patches x feature_dim]

        # Batch Size
        bs, enc_num_patches, dim = x.shape

        # map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # append and unshuffle mask tokens
        bs, num_patch, ch = ids_restore.shape
        # ids_restore: [bs x num_patch x n_vars]
        ids_restore = ids_restore.transpose(1, 2).reshape(bs * ch, num_patch)
        # ids_restore: [bs * n_vars x num_patch]
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )

        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # add positional embedding
        x += self.predictor_pos_embed

        # -- fwd prop
        for blk in self.predictor_blocks:
            x, attn, layer_res_ffn = blk(x)

        if self.layer_norm_first:
            x = self.predictor_norm(x)

        # final projection
        x = self.predictor_proj(x)

        return x


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class DecoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual):
        if residual is None or residual.size(1) != x.size(1):
            return x

        ret = x + residual

        return ret


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Decoder1d(DecoderBase):
    def __init__(
        self,
        input_dim,
        decoder_layers,
        decoder_dim,
        decoder_kernel,
        decoder_groups=1,
        projection_layers=1,
        projection_ratio=2.0,
    ):
        super().__init__()

        def make_block(in_dim):
            block = [
                nn.Conv1d(
                    in_dim,
                    decoder_dim,
                    kernel_size=decoder_kernel,
                    padding=decoder_kernel // 2,
                    groups=decoder_groups,
                ),
                SamePad(decoder_kernel),
                TransposeLast(),
                LayerNorm(decoder_dim, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else decoder_dim)
                for i in range(decoder_layers)
            ]
        )

        projs = []
        curr_dim = decoder_dim
        for i in range(projection_layers - 1):
            next_dim = int(curr_dim * projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x):
        x = x.transpose(1, 2)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual)
            residual = x

        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


class CNNPredictor(nn.Module):
    """Time Series Transformer with channel independence"""

    def __init__(
        self,
        num_patches,
        encoder_embed_dim,
        predictor_embed_dim,
        depth,
        kernel_size,
        mask_noise_std,
        init_std=0.02,
        learn_pe=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = predictor_embed_dim
        # self.predictor_embed = nn.Linear(
        #     encoder_embed_dim, predictor_embed_dim, bias=True
        # )
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # 1d pos embed
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=learn_pe
        )
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1], num_patches, cls_token=False
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )
        self.mask_noise_std = mask_noise_std

        self.decoder = Decoder1d(
            input_dim=encoder_embed_dim,
            decoder_layers=depth,
            decoder_dim=predictor_embed_dim,
            decoder_kernel=kernel_size,
        )

        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ids_restore):
        # x: [bs x enc_num_patches x feature_dim]

        # Batch Size
        # bs, enc_num_patches, dim = x.shape

        # map from encoder-dim to pedictor-dim
        # x = self.predictor_embed(x)

        # append and unshuffle mask tokens
        bs, num_patch, ch = ids_restore.shape
        # ids_restore: [bs x num_patch x n_vars]
        ids_restore = ids_restore.transpose(1, 2).reshape(bs * ch, num_patch)
        # ids_restore: [bs * n_vars x num_patch]

        num_masked = ids_restore.shape[1] - x.shape[1]

        # mask tokens for cnn decoder are randomly sampled from a Gaussian distribution
        mask_tokens = x.new_empty(
            x.size(0),
            num_masked,
            x.size(-1),
        ).normal_(0, self.mask_noise_std)

        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # TODO experiment with positional embedding
        # add positional embedding
        # x += self.predictor_pos_embed

        # -- fwd prop
        x = self.decoder(x)

        return x
