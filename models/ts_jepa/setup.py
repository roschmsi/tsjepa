from models.ts_jepa.tensors import trunc_normal_
from models.ts_jepa.model import (
    TSTClassifier,
    TSTEncoder,
    LinearForecaster,
    PETForecaster,
    TSTPredictor,
)
import torch
import torch.nn as nn
from models.patch_tst.layers.basics import Transpose


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


# def init_model_pretraining(
#     device,
#     seq_len,
#     patch_size,
#     in_chans,
#     enc_embed_dim,
#     enc_depth,
#     enc_num_heads,
#     enc_mlp_ratio,
#     pred_embed_dim,
#     pred_depth,
#     pred_num_heads,
#     pred_mlp_ratio,
#     norm_layer,
#     output_norm,
#     learn_pe,
#     drop_rate=0,
#     attn_drop_rate=0,
#     drop_path_rate=0,
# ):
#     if norm_layer == "BatchNorm":
#         norm_layer = BatchNorm
#     elif norm_layer == "LayerNorm":
#         norm_layer = LayerNorm

#     encoder = TSTEncoder(
#         seq_len=seq_len,
#         patch_size=patch_size,
#         in_chans=in_chans,
#         embed_dim=enc_embed_dim,
#         depth=enc_depth,
#         num_heads=enc_num_heads,
#         mlp_ratio=enc_mlp_ratio,
#         qkv_bias=True,
#         norm_layer=norm_layer,
#         drop_rate=drop_rate,
#         attn_drop_rate=attn_drop_rate,
#         drop_path_rate=drop_path_rate,
#         output_norm=output_norm,
#         learn_pe=learn_pe,
#     )
#     predictor = TSTPredictor(
#         num_patches=encoder.patch_embed.num_patches,
#         encoder_embed_dim=encoder.embed_dim,
#         predictor_embed_dim=pred_embed_dim,
#         depth=pred_depth,
#         num_heads=pred_num_heads,
#         mlp_ratio=pred_mlp_ratio,
#         qkv_bias=True,
#         norm_layer=norm_layer,
#         drop_rate=drop_rate,
#         attn_drop_rate=attn_drop_rate,
#         drop_path_rate=drop_path_rate,
#         output_norm=output_norm,
#         learn_pe=learn_pe,
#     )

#     def init_weights(m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.LayerNorm):
#             torch.nn.init.constant_(m.bias, 0)
#             torch.nn.init.constant_(m.weight, 1.0)

#     for m in encoder.modules():
#         init_weights(m)

#     for m in predictor.modules():
#         init_weights(m)

#     encoder.to(device)
#     predictor.to(device)
#     return encoder, predictor


# def init_classifier(
#     device,
#     seq_len,
#     patch_size,
#     in_chans,
#     enc_embed_dim,
#     enc_depth,
#     enc_num_heads,
#     enc_mlp_ratio,
#     norm_layer,
#     n_classes,
#     head_dropout,
#     learn_pe,
#     output_norm,
#     drop_rate=0,
#     attn_drop_rate=0,
#     drop_path_rate=0,
# ):
#     if norm_layer == "BatchNorm":
#         norm_layer = BatchNorm
#     elif norm_layer == "LayerNorm":
#         norm_layer = LayerNorm

#     classifier = TSTClassifier(
#         seq_len=seq_len,
#         patch_size=patch_size,
#         in_chans=in_chans,
#         enc_embed_dim=enc_embed_dim,
#         enc_depth=enc_depth,
#         enc_num_heads=enc_num_heads,
#         enc_mlp_ratio=enc_mlp_ratio,
#         norm_layer=norm_layer,
#         drop_rate=drop_rate,
#         attn_drop_rate=attn_drop_rate,
#         drop_path_rate=drop_path_rate,
#         n_classes=n_classes,
#         head_dropout=head_dropout,
#         learn_pe=learn_pe,
#         output_norm=output_norm,
#     )

#     def init_weights(m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.LayerNorm):
#             torch.nn.init.constant_(m.bias, 0)
#             torch.nn.init.constant_(m.weight, 1.0)

#     for m in classifier.modules():
#         init_weights(m)

#     classifier.to(device)
#     return classifier


# def init_forecaster(
#     device,
#     seq_len,
#     patch_size,
#     in_chans,
#     enc_embed_dim,
#     enc_depth,
#     enc_num_heads,
#     enc_mlp_ratio,
#     norm_layer,
#     head_dropout,
#     num_patch,
#     forecast_len,
#     learn_pe,
#     head,
#     drop_rate=0,
#     attn_drop_rate=0,
#     drop_path_rate=0,
# ):
#     if norm_layer == "BatchNorm":
#         norm_layer = BatchNorm
#     elif norm_layer == "LayerNorm":
#         norm_layer = LayerNorm

#     if head == "linear":
#         forecaster = LinearForecaster(
#             seq_len=seq_len,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             enc_embed_dim=enc_embed_dim,
#             enc_depth=enc_depth,
#             enc_num_heads=enc_num_heads,
#             enc_mlp_ratio=enc_mlp_ratio,
#             norm_layer=norm_layer,
#             drop_rate=drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=drop_path_rate,
#             head_dropout=head_dropout,
#             num_patch=num_patch,
#             forecast_len=forecast_len,
#             learn_pe=learn_pe,
#         )
#     elif head == "transformer":
#         forecaster = PETForecaster(
#             seq_len=seq_len,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             enc_embed_dim=enc_embed_dim,
#             enc_depth=enc_depth,
#             enc_num_heads=enc_num_heads,
#             enc_mlp_ratio=enc_mlp_ratio,
#             norm_layer=norm_layer,
#             drop_rate=drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=drop_path_rate,
#             head_dropout=head_dropout,
#             num_patch=num_patch,
#             forecast_len=forecast_len,
#             learn_pe=learn_pe,
#         )

#     def init_weights(m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.LayerNorm):
#             torch.nn.init.constant_(m.bias, 0)
#             torch.nn.init.constant_(m.weight, 1.0)

#     for m in forecaster.modules():
#         init_weights(m)

#     forecaster.to(device)
#     return forecaster


# # exclude bias and 1d weights (normalization) from weight decay
# def init_optimizer_enc_pred(encoder, predictor, lr, weight_decay, epochs):
#     param_groups = [
#         {
#             "params": (
#                 p
#                 for n, p in encoder.named_parameters()
#                 if ("bias" not in n) and (len(p.shape) != 1)
#             )
#         },
#         {
#             "params": (
#                 p
#                 for n, p in predictor.named_parameters()
#                 if ("bias" not in n) and (len(p.shape) != 1)
#             )
#         },
#         {
#             "params": (
#                 p
#                 for n, p in encoder.named_parameters()
#                 if ("bias" in n) or (len(p.shape) == 1)
#             ),
#             "WD_exclude": True,
#             "weight_decay": 0,
#         },
#         {
#             "params": (
#                 p
#                 for n, p in predictor.named_parameters()
#                 if ("bias" in n) or (len(p.shape) == 1)
#             ),
#             "WD_exclude": True,
#             "weight_decay": 0,
#         },
#     ]

#     # param_groups = [*encoder.parameters(), *predictor.parameters()]

#     optimizer = torch.optim.AdamW(
#         param_groups,
#         lr=lr,
#         weight_decay=weight_decay,
#     )

#     return optimizer


def init_optimizer(model, lr, weight_decay, epochs):
    param_groups = [
        {
            "params": (
                p
                for n, p in model.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in model.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )

    return optimizer


def init_scheduler(optimizer, config):
    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.epochs
        )
    elif config.scheduler == "CosineAnnealingLRWithWarmup":
        scheduler_1 = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=config.start_factor,
            total_iters=config.warmup,
        )
        scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.epochs - config.warmup
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_1, scheduler_2],
            milestones=[config.warmup],
        )
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=config.step_size, gamma=0.1
        )

    else:
        scheduler = None

    return scheduler
