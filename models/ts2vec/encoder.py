import torch
import torch.nn as nn
import math
from models.ts_jepa.model import get_1d_sincos_pos_embed
from models.ts_jepa.mask import apply_masks
from models.ts_jepa.tensors import trunc_normal_
from models.patch_tst.layers.basics import Transpose


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


def get_activation_fn(activation):
    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


class PatchEmbed(nn.Module):
    """1D Time Series to Patch Embedding"""

    def __init__(
        self,
        ts_length: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ):
        super().__init__()

        self.num_patches = ts_length // patch_size

        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x


# from I-JEPA
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
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        norm_layer,
        layer_norm_first,
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
        self.self_attn_layer_norm = norm_layer(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = norm_layer(self.embedding_dim)

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


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(dim),
            Transpose(1, 2),
        )

    def forward(self, x):
        return self.norm(x)


# based on I-JEPA
class TransformerEncoder(nn.Module):
    """Time Series Transformer with channel independence"""

    def __init__(
        self,
        seq_len,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        activation="gelu",
        activation_drop_rate=0.0,
        norm="LayerNorm",
        init_std=0.02,
        layer_norm_first=True,
        learn_pe=False,
        use_mask_tokens=False,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.layer_norm_first = layer_norm_first

        self.patch_embed = nn.Linear(patch_size, embed_dim)

        num_patches = int(seq_len // patch_size)

        # norm layer
        if norm == "LayerNorm":
            norm_layer = LayerNorm
        elif norm == "BatchNorm":
            norm_layer = BatchNorm
        else:
            raise NotImplementedError(f"Norm type {norm} not supported")

        # 1d pos embed
        # absolute position encoding
        # TODO add option for relative position encoding
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
                    norm_layer=norm_layer,
                    layer_norm_first=layer_norm_first,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.self_attn.proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        # x: [bs x seq_len x n_vars]

        # patchify x
        # x: [bs x n_vars x seq_len]
        x = self.patch_embed(x)
        # x: [bs x num_patches x embed_dim]

        # add positional embedding to x
        # TODO potentially add input dropout
        pos_embed = self.pos_embed.repeat_interleave(dim=0, repeats=x.shape[0])
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat_interleave(dim=-1, repeats=self.embed_dim)
            pos_embed = pos_embed[mask]
            pos_embed = pos_embed.reshape(x.shape[0], x.shape[1], self.embed_dim)

        x += pos_embed

        # TODO interpolation only for finetuning if necessary
        # x = x + self.interpolate_pos_encoding(x, self.pos_embed)

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

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        pos_embed = pos_embed.transpose(1, 2)
        pos_embed = nn.functional.interpolate(
            pos_embed,
            scale_factor=npatch / N,
            mode="linear",
        )
        pos_embed = pos_embed.transpose(1, 2)
        return pos_embed
