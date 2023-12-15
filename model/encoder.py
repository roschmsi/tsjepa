# Reference: https://github.com/facebookresearch/ijepa

import torch
import torch.nn as nn
from model.positional_encoding import get_1d_sincos_pos_embed


def get_activation_fn(activation):
    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float,
        ffn_embedding_dim: float,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
        layer_norm_first: bool,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            dim=self.embedding_dim,
            num_heads=num_attention_heads,
            proj_drop=attention_dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(x)
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, layer_result


class TransformerEncoder(nn.Module):
    """Time Series Transformer with channel independence"""

    def __init__(
        self,
        patch_size,
        num_patch,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
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
        self.embed_dim = embed_dim
        self.layer_norm_first = layer_norm_first
        self.patch_embed = nn.Linear(patch_size, embed_dim)

        num_patches = num_patch

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=learn_pe
        )
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], num_patches, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=int(embed_dim * mlp_ratio),
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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_kept=None):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed

        if ids_kept is not None:
            pos_embed = pos_embed.repeat_interleave(dim=0, repeats=x.shape[0])
            pos_embed = torch.gather(
                pos_embed, dim=1, index=ids_kept.repeat(1, 1, self.embed_dim)
            )

        x += pos_embed

        # layer results after ffn in every block
        layer_results_ffn = []
        layer_results = []

        # fwd prop
        for i, blk in enumerate(self.blocks):
            x, attn, layer_res_ffn = blk(x)
            layer_results.append(x)
            layer_results_ffn.append(layer_res_ffn)

        if self.layer_norm_first:
            x = self.norm(x)

        return {
            "encoder_out": x,
            "encoder_states": layer_results,
            "encoder_states_ffn": layer_results_ffn,
        }
