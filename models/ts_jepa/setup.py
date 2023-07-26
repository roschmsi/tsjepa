from functools import partial
from models.ts_jepa.tensors import trunc_normal_
from models.ts_jepa.model import TSTEncoder, TSTPredictor
import torch
import torch.nn as nn


def init_model(
    device,
    seq_len,
    patch_size,
    in_chans,
    enc_embed_dim,
    enc_depth,
    enc_num_heads,
    enc_mlp_ratio,
    pred_depth,
    pred_embed_dim,
    pred_num_heads,
    pred_mlp_ratio,
    drop_rate=0,
    attn_drop_rate=0,
):
    encoder = TSTEncoder(
        seq_len=seq_len,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=enc_embed_dim,
        depth=enc_depth,
        num_heads=enc_num_heads,
        mlp_ratio=enc_mlp_ratio,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )
    predictor = TSTPredictor(
        num_patches=encoder.patch_embed.num_patches,
        encoder_embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads,
        mlp_ratio=pred_mlp_ratio,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    # iterations_per_epoch,
    # start_lr,
    # ref_lr,
    # warmup,
    # num_epochs,
    # wd=1e-6,
    # final_wd=1e-6,
    # final_lr=0.0,
    # use_bfloat16=False,
    # ipe_scale=1.25,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups)
    # scheduler = WarmupCosineSchedule(
    #     optimizer,
    #     warmup_steps=int(warmup * iterations_per_epoch),
    #     start_lr=start_lr,
    #     ref_lr=ref_lr,
    #     final_lr=final_lr,
    #     T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    # )
    # wd_scheduler = CosineWDSchedule(
    #     optimizer,
    #     ref_wd=wd,
    #     final_wd=final_wd,
    #     T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    # )
    # scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer
