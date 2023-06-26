from dataclasses import dataclass


from omegaconf import II
from typing import Optional


@dataclass
class DecoderConfig:
    decoder_dim: int = 64
    decoder_groups: int = 1
    decoder_kernel: int = 7
    decoder_layers: int = 2
    input_dropout: float = 0

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0


@dataclass
class TimeSeriesConfig:
    input_size: int = 512
    in_chans: int = 1
    patch_size: int = 16
    embed_dim: int = 64

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False

    prenet_depth: int = 0
    prenet_layerdrop: float = 0
    prenet_dropout: float = 0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0

    num_extra_tokens: int = 0
    init_extra_token_zero: bool = False

    mask_noise_std: float = 0.01  # for "mask tokens"
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.7
    inverse_mask: bool = True
    mask_prob_adjust: float = 0.07
    keep_masked_pct: float = 0

    mask_length: int = 3
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True

    mask_channel_prob: float = 0.0
    mask_channel_length: int = 4

    ema_local_encoder: bool = True  # used in data2vec_multi
    local_grad_mult: float = 1.0

    use_alibi_encoder: bool = False
    alibi_scale: float = 1.0
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False

    num_alibi_heads: int = II("model.num_heads")
    model_depth: int = II("model.depth")

    decoder: Optional[DecoderConfig] = DecoderConfig()


@dataclass
class TS2VecConfig:
    # beta for smooth l1 loss. 0 means use l2 loss
    loss_beta: float = 0
    # scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)
    loss_scale: Optional[float] = None

    depth: int = 4
    num_heads: int = 1
    embed_dim: int = 64
    mlp_ratio: float = 2

    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0

    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_norm_first: bool = False

    encoder_dropout: float = 0.0
    post_mlp_drop: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0

    average_top_k_layers: int = 2

    end_of_block_targets: bool = False

    clone_batch: int = 4

    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = 0.9
    ema_end_decay: float = 0.9999
    ema_same_dtype: bool = True
    log_norms: bool = True

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = 544000
    max_update: int = 544000  # epochs * len(train_loader)

    # whether to momentum update only the shared transformer encoder
    ema_encoder_only: bool = True

    modality: Optional[TimeSeriesConfig] = TimeSeriesConfig()

    shared_decoder: Optional[DecoderConfig] = None

    mae_init: bool = False
    seed = 42

    skip_ema: bool = False

    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    decoder_group: bool = False
