import torch
import torch.nn as nn

from model.encoder import TransformerEncoderLayer
from model.positional_encoding import get_1d_sincos_pos_embed


def get_predictor(config, max_seq_len):
    if config.predictor == "linear" and config.input_space:
        predictor = nn.Linear(config.enc_d_model, config.patch_len)
    elif config.predictor == "transformer" and config.input_space:
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
    """
    TS-JEPA Transformer Predictor
    Adapted from https://github.com/facebookresearch/ijepa
    """

    def __init__(
        self,
        num_patches,
        encoder_embed_dim,
        predictor_embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        target_dim,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        activation="gelu",
        activation_drop_rate=0.0,
        norm="LayerNorm",
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
        bs, enc_num_patches, dim = x.shape

        # map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # append and unshuffle mask tokens
        bs, num_patch, ch = ids_restore.shape
        ids_restore = ids_restore.transpose(1, 2).reshape(bs * ch, num_patch)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # add positional embedding
        x += self.predictor_pos_embed

        # fwd prop
        for blk in self.predictor_blocks:
            x, attn, layer_res_ffn = blk(x)

        if self.layer_norm_first:
            x = self.predictor_norm(x)

        # final projection
        x = self.predictor_proj(x)

        return x
