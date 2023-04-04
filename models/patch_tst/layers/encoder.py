from typing import Optional
from torch import nn
from torch import Tensor
from models.patch_tst.layers.basics import Transpose, get_activation_fn
from models.patch_tst.layers.attention import MultiheadAttention


class TSTEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
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
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
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

    def forward(self, src: Tensor, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output, prev=scores, key_padding_mask=key_padding_mask
                )
            return output
        else:
            for mod in self.layers:
                output = mod(output)
            return output


class TSTEncoderLayer(nn.Module):
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

        # Multi-Head attention
        self.res_attention = res_attention
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
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

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

    def forward(
        self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask=None
    ):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src, src, src, prev, key_padding_mask=key_padding_mask
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask
            )
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
