# Reference: https://github.com/yuqinie98/PatchTST

from typing import Optional
from torch import nn
from torch import Tensor
from models.patch_tst.layers.basics import Transpose, get_activation_fn
from models.patch_tst.layers.attention import (
    MultiheadAttention,
    TemporalMultiheadAttention,
)
import torch


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


class TSTSignalTimeEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        d_temp,
        d_ff,
        dropout,
        norm="BatchNorm",
        pre_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        res_attention=False,
        store_attn=False,
        use_time_features=False,
    ):
        super().__init__()
        layers = []

        for _ in range(num_layers):
            # layers.append(
            #     TemporalAttentionLayer(
            #         num_heads=num_heads,
            #         d_model=d_model,
            #         d_temp=d_temp,
            #         d_ff=d_ff,
            #         dropout=dropout,
            #         norm=norm,
            #         pre_norm=pre_norm,
            #         activation=activation,
            #         attn_dropout=attn_dropout,
            #         res_attention=res_attention,
            #         store_attn=store_attn,
            #     )
            # )
            layers.append(
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
            )

        self.layers = nn.ModuleList(layers)
        self.res_attention = res_attention

    def forward(self, src: Tensor, temp_enc: Tensor, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        # TODO investigate usefulness of res_attention
        if self.res_attention:
            for mod in self.layers:
                if type(mod) == TSTEncoderLayer:
                    output, scores = mod(
                        output, prev=scores, key_padding_mask=key_padding_mask
                    )
                elif type(mod) == TemporalAttentionLayer:
                    output, _ = mod(
                        output, temp_enc, prev=scores, key_padding_mask=key_padding_mask
                    )
                else:
                    raise ValueError("Unknown layer type")
            return output
        else:
            for mod in self.layers:
                if type(mod) == TSTEncoderLayer:
                    output = mod(output)
                elif type(mod) == TemporalAttentionLayer:
                    output = mod(output, temp_enc)
                else:
                    raise ValueError("Unknown layer type")
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


# for different temporal encoding per unit
class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        d_model,
        d_temp,
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
        # assert (
        #     not d_model % num_heads
        # ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        # d_k = d_model // num_heads
        # d_v = d_model // num_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn_hour = TemporalMultiheadAttention(
            d_model=d_model,
            d_temp=d_temp,
            num_heads=1,
            d_k=d_temp,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.self_attn_weekday = TemporalMultiheadAttention(
            d_model=d_model,
            d_temp=d_temp,
            num_heads=1,
            d_k=d_temp,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.self_attn_day = TemporalMultiheadAttention(
            d_model=d_model,
            d_temp=d_temp,
            num_heads=1,
            d_k=d_temp,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.self_attn_month = TemporalMultiheadAttention(
            d_model=d_model,
            d_temp=d_temp,
            num_heads=1,
            d_k=d_temp,
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

        self.temp_scores = nn.Parameter(torch.ones(4))

    def forward(
        self,
        src: Tensor,
        temp: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask=None,
    ):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
            # Multi-Head attention

        # if self.res_attention:
        #     src2, attn, scores = self.self_attn(
        #         Q=temp, K=temp, V=src, prev=prev, key_padding_mask=key_padding_mask
        #     )
        # else:

        # hour
        temp_hour = temp[:, :, :24]
        src2_hour, _ = self.self_attn_hour(
            Q=temp_hour, K=temp_hour, V=src, key_padding_mask=key_padding_mask
        )

        # weekday
        temp_weekday = temp[:, :, 24 : 24 + 7]
        src2_weekday, _ = self.self_attn_weekday(
            Q=temp_weekday, K=temp_weekday, V=src, key_padding_mask=key_padding_mask
        )

        # day
        temp_day = temp[:, :, 24 + 7 : 24 + 7 + 31]
        src2_day, _ = self.self_attn_day(
            Q=temp_day, K=temp_day, V=src, key_padding_mask=key_padding_mask
        )

        # month
        temp_month = temp[:, :, 24 + 7 + 31 :]
        src2_month, _ = self.self_attn_month(
            Q=temp_month, K=temp_month, V=src, key_padding_mask=key_padding_mask
        )

        # weights = torch.nn.functional.softmax(self.temp_scores)

        # src2 = (
        #     weights[0] * src2_hour
        #     + weights[1] * src2_weekday
        #     + weights[2] * src2_day
        #     + weights[3] * src2_month
        # )

        # TODO fix store attention
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = (
            src
            # + self.dropout_attn(src2_hour)
            + 0.25 * self.dropout_attn(src2_weekday)
            # + self.dropout_attn(src2_day)
            # + self.dropout_attn(src2_month)
        )
        # Add: residual connection with residual dropout
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


# # for full temporal encoding
# class TemporalAttentionLayer(nn.Module):
#     def __init__(
#         self,
#         num_heads,
#         d_model,
#         d_temp,
#         d_ff,
#         dropout,
#         norm="BatchNorm",
#         pre_norm=False,
#         activation="gelu",
#         attn_dropout=0.0,
#         res_attention=False,
#         store_attn=False,
#         bias=True,
#     ):
#         super().__init__()
#         # assert (
#         #     not d_model % num_heads
#         # ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
#         # d_k = d_model // num_heads
#         # d_v = d_model // num_heads

#         # Multi-Head attention
#         self.res_attention = res_attention
#         self.self_attn = TemporalMultiheadAttention(
#             d_model=d_model,
#             d_temp=d_temp,
#             num_heads=1,
#             d_k=d_temp,
#             attn_dropout=attn_dropout,
#             proj_dropout=dropout,
#             res_attention=res_attention,
#         )

#         # Add & Norm
#         self.dropout_attn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_attn = nn.Sequential(
#                 Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
#             )
#         else:
#             self.norm_attn = nn.LayerNorm(d_model)

#         # Position-wise Feed-Forward
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_ff, bias=bias),
#             get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model, bias=bias),
#         )

#         # Add & Norm
#         self.dropout_ffn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_ffn = nn.Sequential(
#                 Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
#             )
#         else:
#             self.norm_ffn = nn.LayerNorm(d_model)

#         self.pre_norm = pre_norm
#         self.store_attn = store_attn

#     def forward(
#         self,
#         src: Tensor,
#         temp: Tensor,
#         prev: Optional[Tensor] = None,
#         key_padding_mask=None,
#     ):
#         """
#         src: tensor [bs x q_len x d_model]
#         """
#         # Multi-Head attention sublayer
#         if self.pre_norm:
#             src = self.norm_attn(src)
#         # Multi-Head attention

#         if self.res_attention:
#             src2, attn, scores = self.self_attn(
#                 Q=temp, K=temp, V=src, prev=prev, key_padding_mask=key_padding_mask
#             )
#         else:
#             src2, attn = self.self_attn(
#                 Q=temp, K=temp, V=src, key_padding_mask=key_padding_mask
#             )

#         if self.store_attn:
#             self.attn = attn
#         # Add & Norm
#         src = src + self.dropout_attn(
#             src2
#         )  # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_attn(src)

#         # Feed-forward sublayer
#         if self.pre_norm:
#             src = self.norm_ffn(src)
#         # Position-wise Feed-Forward
#         src2 = self.ff(src)
#         # Add & Norm
#         src = src + self.dropout_ffn(
#             src2
#         )  # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_ffn(src)

#         if self.res_attention:
#             return src, scores
#         else:
#             return src
