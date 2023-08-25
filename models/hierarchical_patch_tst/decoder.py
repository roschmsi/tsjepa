# Reference: https://github.com/yuqinie98/PatchTST

from typing import Optional
from torch import nn
from torch import Tensor
from models.patch_tst.layers.basics import Transpose, get_activation_fn
from models.patch_tst.layers.attention import MultiheadAttention
from models.patch_tst.layers.pos_encoding import positional_encoding


class TSTDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        num_patch,
        norm="BatchNorm",
        pre_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        res_attention=False,
        store_attn=False,
    ):
        super().__init__()

        self.W_pos = positional_encoding(
            pe="sincos", learn_pe=False, q_len=num_patch, d_model=d_model
        )

        self.layers = nn.ModuleList(
            [
                TSTDecoderLayer(
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
                for i in range(num_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(self, x, cross, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # add positional encoding to start token
        cross_len = cross.shape[1]
        cross = cross + self.W_pos[:cross_len, :]
        x = x + self.W_pos[cross_len:, :]

        # TODO residual attention optional
        for layer in self.layers:
            x = layer(x=x, cross=cross)

        return x


class TSTDecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        d_model,
        d_ff,
        dropout,
        norm="BatchNorm",
        pre_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        res_attention=False,
        store_attn=False,
        bias=True,
    ):
        super().__init__()
        assert (
            not d_model % num_heads
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        d_k = d_model // num_heads
        d_v = d_model // num_heads

        # self attention
        self.self_attn = MultiheadAttention(
            d_model,
            num_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_self_attn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_self_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_self_attn = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MultiheadAttention(
            d_model,
            num_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_cross_attn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_cross_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_cross_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, x, cross, prev: Optional[Tensor] = None, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # self attention
        x2, attn = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)

        # Add & Norm
        x = x + self.dropout_self_attn(x2)
        x = self.norm_self_attn(x)

        # cross attention
        x2, attn = self.cross_attn(x, cross, cross, key_padding_mask=key_padding_mask)

        # Add & Norm
        x = x + self.dropout_cross_attn(x2)
        x = self.norm_cross_attn(x)

        # Position-wise Feed-Forward
        x2 = self.ff(x)

        # Add & Norm
        x = x + self.dropout_ffn(x2)  # Add: residual connection with residual dropout
        x = self.norm_ffn(x)

        return x
