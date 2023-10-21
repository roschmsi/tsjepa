from functools import partial
from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from models.patch_tst.layers.pos_encoding import PositionalEncoding
from models.ts2vec.config import TimeSeriesConfig
from models.ts2vec.decoder import Decoder1d
from models.ts2vec.utils import (
    BlockEncoder,
    FixedPositionalEncoder,
    GradMultiply,
    MaskInfo,
    MaskSeed,
    TransformerDecoder,
    compute_mask_indices,
    index_put,
)


class ModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        modality_cfg,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: nn.Module,
        get_alibi_bias: Callable[[int, int, int, int, int], torch.Tensor],
    ):
        super().__init__()

        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = decoder
        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None

        self.local_grad_mult = self.modality_cfg.local_grad_mult

        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim)
            )
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])

        self.alibi_scale = None
        # if self.get_alibi_bias is not None:
        #     self.alibi_scale = nn.Parameter(
        #         torch.full(
        #             (
        #                 (modality_cfg.prenet_depth + modality_cfg.model_depth)
        #                 if modality_cfg.learned_alibi_scale_per_layer
        #                 else 1,
        #                 1,
        #                 self.modality_cfg.num_alibi_heads
        #                 if modality_cfg.learned_alibi_scale_per_head
        #                 else 1,
        #                 1,
        #                 1,
        #             ),
        #             modality_cfg.alibi_scale,
        #             dtype=torch.float,
        #         ),
        #         requires_grad=modality_cfg.learned_alibi_scale,
        #     )

        # if modality_cfg.learned_alibi and self.get_alibi_bias is not None:
        #     assert modality_cfg.alibi_max_pos is not None
        #     alibi_bias = self.get_alibi_bias(
        #         batch_size=1,
        #         time_steps=modality_cfg.alibi_max_pos,
        #         heads=modality_cfg.num_alibi_heads,
        #         scale=1.0,
        #         dtype=torch.float,
        #         device="cpu",
        #     )
        #     self.alibi_bias = nn.Parameter(alibi_bias)
        #     self.get_alibi_bias = partial(
        #         _learned_alibi_bias, alibi_bias=self.alibi_bias
        #     )

    # def upgrade_state_dict_named(self, state_dict, name):
    #     k = f"{name}.alibi_scale"
    #     if k in state_dict and state_dict[k].dim() == 4:
    #         state_dict[k] = state_dict[k].unsqueeze(0)

    #     return state_dict

    # def convert_padding_mask(self, x, padding_mask):
    #     return padding_mask

    def decoder_input(self, x, mask_info: MaskInfo):
        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        num_extra = self.modality_cfg.num_extra_tokens

        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.modality_cfg.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)

            if self.modality_cfg.decoder.add_positions_masked:
                assert self.fixed_positional_encoder is not None
                pos = self.fixed_positional_encoder(x, None)
                x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, num_extra:]

        if self.modality_cfg.decoder.add_positions_all:
            assert self.fixed_positional_encoder is not None
            x = x + self.fixed_positional_encoder(x, None)

        return x, mask_info

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(features)

        x = self.project_features(x)
        return x

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        # if padding_mask is not None:
        #     padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        # if mask and clone_batch == 1:
        #     local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)

        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                # if mask_seeds is not None:
                #     clone_hash = [
                #         int(hash((mask_seeds.seed, ind)) % 1e10)
                #         for ind in range(clone_batch - 1)
                #     ]
                #     clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                #     id = mask_seeds.ids
                #     id = id.repeat_interleave(clone_batch, 0)
                #     id = id.view(-1, clone_batch) + clone_hash.to(id)
                #     id = id.view(-1)
                #     mask_seeds = MaskSeed(
                #         seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                #     )
                # if padding_mask is not None:
                #     padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        # if self.relative_positional_encoder is not None:
        #     x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            # if x_pos is not None:
            #     x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        alibi_bias = None
        alibi_scale = self.alibi_scale

        # if self.get_alibi_bias is not None:
        #     alibi_bias = self.get_alibi_bias(
        #         batch_size=pre_mask_B,
        #         time_steps=orig_T,
        #         heads=self.modality_cfg.num_alibi_heads,
        #         dtype=torch.float32,
        #         device=x.device,
        #     )

        #     if alibi_scale is not None:
        #         alibi_scale = alibi_scale.clamp_min(0)
        #         if alibi_scale.size(0) == 1:
        #             alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
        #             alibi_scale = None

        #     if clone_batch > 1:
        #         alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

        #     if mask_info is not None and remove_masked:
        #         alibi_bias = masked_alibi(alibi_bias, mask_info)

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            # if masked_padding_mask is not None:
            #     # B x T
            #     masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            # if alibi_bias is not None:
            #     # B x H x T x T
            #     alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.modality_cfg.prenet_depth]
            if alibi_scale is not None
            else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale[self.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

    def reset_parameters(self):
        pass

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg

            mask_prob = cfg.mask_prob

            if (
                cfg.mask_prob_min is not None
                and cfg.mask_prob_min >= 0
                and cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)

            if mask_prob > 0:
                if cfg.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob

                    mask = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        mask_prob,
                        cfg.mask_length,
                        min_masks=1,
                        require_same_masks=True,
                        mask_dropout=cfg.mask_dropout,
                        add_masks=cfg.add_masks,
                        seed=mask_seed.seed if mask_seed is not None else None,
                        epoch=mask_seed.update if mask_seed is not None else None,
                        indices=mask_seed.ids if mask_seed is not None else None,
                    )

                    mask = torch.from_numpy(mask).to(device=x.device)
                    if self.modality_cfg.inverse_mask:
                        mask = 1 - mask
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        len_keep = T - mask[0].sum()
        if self.modality_cfg.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )
        return mask_info

    def apply_mask(self, x, mask_info):
        cfg = self.modality_cfg
        B, T, C = x.shape

        if mask_info is not None:
            mask = mask_info.mask
            if cfg.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(
                    0, cfg.mask_noise_std
                )
                x = index_put(x, mask, masks)
        if cfg.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                (B, C),
                None,
                cfg.mask_channel_prob,
                cfg.mask_channel_length,
            )
            mask_channel = (
                torch.from_numpy(mask_channel)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x = index_put(x, mask_channel, 0)
        return x

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None


class TimeSeriesPatchEmbed(nn.Module):
    """1D Time Series to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 64,
        bias: bool = True,
    ):
        super().__init__()

        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x


class TimeSeriesEncoder(ModalitySpecificEncoder):
    modality_cfg: TimeSeriesConfig

    def __init__(
        self,
        modality_cfg: TimeSeriesConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task=None,
    ):
        ts_size = modality_cfg.input_size  # time series length
        patch_size = modality_cfg.patch_size  # time series patch length
        num_patches = ts_size // patch_size  # number of patches

        # generate and initialize local encoder (patch embedding)
        local_encoder = TimeSeriesPatchEmbed(
            patch_size=modality_cfg.patch_size,
            in_chans=modality_cfg.in_chans,
            embed_dim=modality_cfg.embed_dim,
        )

        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # if modality_cfg.embed_dim != embed_dim:
        #     local_encoder = nn.Sequential(
        #         local_encoder,
        #         nn.Linear(modality_cfg.embed_dim, embed_dim),
        #     )

        project_features = nn.Identity()

        emb = nn.Parameter(PositionalEncoding(q_len=num_patches, d_model=embed_dim))
        # pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
        fixed_positional_encoder = (
            FixedPositionalEncoder(emb) if modality_cfg.fixed_positions else None
        )

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        # transformer encoder for context
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        # get mae decoder
        if modality_cfg.transformer_decoder:
            # if modality_cfg.enc_dec_transformer:
            #     decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            # else:
            dec_enc = BlockEncoder(
                nn.ModuleList(
                    make_block(0, modality_cfg.decoder.decoder_dim, 8)
                    for _ in range(modality_cfg.decoder.decoder_layers)
                ),
                None,
                layer_norm_first,
                0,
                0,
            )
            decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = (
                Decoder1d(modality_cfg.decoder, embed_dim)
                if modality_cfg.decoder is not None
                else None
            )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        shape=None,
        precomputed_mask=None,
    ):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        # else:
        #     from fairseq.data.data_utils import compute_block_mask_2d

        #     if shape is not None:
        #         B, L, D = shape
        #     else:
        #         B, L, D = x.shape

        #     mask = compute_block_mask_2d(
        #         shape=(B, L),
        #         mask_prob=self.modality_cfg.mask_prob,
        #         mask_length=self.modality_cfg.mask_length,
        #         mask_prob_adjust=self.modality_cfg.mask_prob_adjust,
        #         inverse_mask=self.modality_cfg.inverse_mask,
        #         require_same_masks=True,
        #         mask_dropout=self.modality_cfg.mask_dropout,
        #     )

        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def decoder_input(self, x, mask_info):
        if (
            not self.modality_cfg.transformer_decoder
            or not self.modality_cfg.enc_dec_transformer
        ):
            return super().decoder_input(x, mask_info)

        # inp_drop = self.modality_cfg.decoder.input_dropout
        # if inp_drop > 0:
        #     x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        # kv = x[:, self.modality_cfg.num_extra_tokens :]

        # assert self.fixed_positional_encoder is not None
        # pos = self.fixed_positional_encoder(x, None).expand(x.size(0), -1, -1)

        # mask = mask_info.mask.bool()
        # if self.modality_cfg.decoder.add_positions_all:
        #     kv = kv + pos[~mask].view(kv.shape)

        # q = pos[mask].view(x.size(0), -1, x.size(-1))

        # return q, kv

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        # channel independence
        B, L, C = features.shape
        features = features.transpose(1, 2).reshape(-1, 1, L)
        # features: [batch_size * num_channels, 1, seq_len]
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )


def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


# adapted from MAE
def random_masking(x, mask_ratio, mask_seed: Optional[MaskSeed]):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(N, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


# def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
#     return torch.gather(
#         x,
#         dim=1,
#         index=mask_info.ids_keep,
#     )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def get_alibi_bias(
    alibi_biases,
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
            .to(dtype=dtype, device=device)
            .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def _learned_alibi_bias(
    alibi_bias,
    batch_size,
    time_steps,
    heads,
    scale,
    dtype,
    device,
):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device

    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode="replicate")

    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias
