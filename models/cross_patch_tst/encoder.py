import torch.nn as nn
from torch import Tensor

from models.cross_patch_tst.attn import TwoStageAttentionLayer


class CrossEncoder(nn.Module):
    def __init__(
        self,
        num_patch,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        factor,
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
                TwoStageAttentionLayer(
                    seg_num=num_patch,
                    factor=factor,
                    d_model=d_model,
                    n_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, key_padding_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src

        for mod in self.layers:
            output = mod(output)

        return output
