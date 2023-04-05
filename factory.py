from functools import partial

import torch

from data.dataset import (
    ClassificationDataset,
    ClassificationPatchDataset,
    ForecastingPatchDataset,
    PretrainingDataset,
    PretrainingPatchDataset,
    collate_patch_superv,
    collate_patch_unsuperv,
    collate_superv,
    collate_unsuperv,
)
from models.cnn_transformer.model import CNNClassifier, CNNTransformer
from models.fedformer.model import (
    CNNDecompTimeFreqEncoder,
    CNNFEDformerEncoder,
    CNNTimeFreqEncoder,
    DecompFEDformerEncoder,
    FEDformerEncoder,
)
from models.masked_autoencoder.model import (
    MaskedAutoencoder,
    MaskedAutoencoderPredictor,
)
from models.masked_autoencoder.pretraining_masked_autoencoder_search_space import (
    get_pretraining_masked_autoencoder_search_space,
)
from models.masked_autoencoder_tc.model import (
    MaskedAutoencoderTC,
    MaskedAutoencoderTCPredictor,
)
from models.masked_autoencoder_t.model import (
    MaskedAutoencoderT,
    MaskedAutoencoderTPredictor,
)
from models.patch_tst.model import PatchTST
from models.patch_tst.patch_tst_search_space import get_patch_tst_search_space
from models.patch_tst_t.model import PatchTransformerT
from models.patch_tst_tc.model import PatchTransformerTC
from models.transformer.model import (
    TSTransformerEncoder,
    TSTransformerEncoderClassifier,
)
from runner import (
    ForecastingRunner,
    SupervisedRunner,
    UnsupervisedPatchRunner,
    UnsupervisedRunner,
)


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    if config.task == "pretraining":
        if config.use_patch:
            dataset = partial(
                PretrainingPatchDataset,
                masking_ratio=config.masking_ratio,
                patch_len=config.patch_len,
                stride=config.stride,
                debug=config.debug,
                only_time_masking=True
                if (
                    config.model_name == "masked_autoencoder_t"
                    or config.model_name == "patch_tst_t"
                )
                else False,
            )
            return (
                dataset,
                partial(
                    collate_patch_unsuperv,
                    feat_dim=config.feat_dim,
                    patch_len=config.patch_len,
                    stride=config.stride,
                    masking_ratio=config.masking_ratio,
                ),
                partial(
                    UnsupervisedPatchRunner,
                    mae=config.mae,
                ),
            )
        else:
            return (
                partial(
                    PretrainingDataset,
                    masking_ratio=config.masking_ratio,
                    mean_mask_length=config.mean_mask_length,
                ),
                collate_unsuperv,
                UnsupervisedRunner,
            )

    elif config.task == "classification":
        if config.use_patch:
            return (
                partial(
                    ClassificationPatchDataset,
                    patch_len=config.patch_len,
                    stride=config.stride,
                ),
                partial(
                    collate_patch_superv,
                    patch_len=config.patch_len,
                    stride=config.stride,
                    masking_ratio=config.masking_ratio,
                ),
                SupervisedRunner,
            )
        else:
            return ClassificationDataset, collate_superv, SupervisedRunner

    elif config.task == "forecasting":
        if config.use_patch:
            return (
                partial(
                    ForecastingPatchDataset,
                    patch_len=config.patch_len,
                    stride=config.stride,
                ),
                partial(
                    collate_patch_superv,
                    patch_len=config.patch_len,
                    stride=config.stride,
                    masking_ratio=config.masking_ratio,
                ),
                ForecastingRunner,
            )
        else:
            return ClassificationDataset, collate_superv, SupervisedRunner

    else:
        raise NotImplementedError("Task '{}' not implemented".format(config["task"]))


def model_factory(config):
    if "seq_len" in config.keys():
        max_seq_len = config.seq_len
    else:
        max_seq_len = config.window * config.fs

    if config.use_patch:
        num_patch = (max_seq_len - config.patch_len) // config.stride + 1

    if config.task == "pretraining":
        c_out = config.feat_dim
    elif config.task == "classification":
        c_out = config.num_classes
    elif config.task == "forecasting":
        c_out = config.pred_len
    else:
        raise ValueError("task not specified")

    # end-to-end supervised models
    if config.model_name == "cnn_encoder":
        return CNNClassifier(
            feat_dim=config.feat_dim,
            d_model=config.d_model,
            num_classes=config.num_classes,
            num_cnn=config.num_cnn,
        )
    elif config.model_name == "cnn_transformer":
        return CNNTransformer(
            feat_dim=config.feat_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_classes=config.num_classes,
            max_seq_len=max_seq_len,
            dropout=config.dropout,
            num_cnn=config.num_cnn,
            cls_token=config.cls_token,
        )
    elif config.model_name == "fedformer_encoder":
        config.activation = "relu"
        return FEDformerEncoder(config)
    elif config.model_name == "cnn_fedformer_encoder":
        return CNNFEDformerEncoder(config)
    elif config.model_name == "decomp_fedformer_encoder":
        return DecompFEDformerEncoder(config)
    elif config.model_name == "cnn_time_freq_encoder":
        return CNNTimeFreqEncoder(config)
    elif config.model_name == "cnn_decomp_time_freq_encoder":
        return CNNDecompTimeFreqEncoder(config)
    # transformer
    elif config.model_name == "transformer":
        if config.task == "pretraining":
            return TSTransformerEncoder(
                feat_dim=config.feat_dim,
                max_len=max_seq_len,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm=config.norm,
                activation="relu",
                pos_encoding="fixed",
                freeze=config.freeze,
            )
        elif config.task == "classification":
            return TSTransformerEncoderClassifier(
                feat_dim=config.feat_dim,
                max_len=max_seq_len,
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm=config.norm,
                activation="relu",
                pos_encoding="fixed",
                num_classes=config.num_classes,
                freeze=config.freeze,
            )
    # patch tst with channel independence
    elif config.model_name == "patch_tst":
        return PatchTST(
            c_in=config.feat_dim,
            c_out=c_out,
            num_patch=num_patch,
            patch_len=config.patch_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            shared_embedding=config.shared_embedding,
            norm=config.norm,
            activation=config.activation,
            pe="sincos",
            learn_pe=config.learn_pe,
            ch_token=config.ch_token,
            cls_token=config.cls_token,
            task=config.task,
            head_dropout=config.head_dropout,
        )
    # patch tst with temporal encoding
    elif config.model_name == "patch_tst_t":
        return PatchTransformerT(
            c_in=config.feat_dim,
            c_out=c_out,
            num_patch=num_patch,
            patch_len=config.patch_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            shared_embedding=config.shared_embedding,
            norm=config.norm,
            activation=config.activation,
            pe="sincos",
            learn_pe=config.learn_pe,
            cls_token=config.cls_token,
            task=config.task,
            head_dropout=config.head_dropout,
        )
    # patch tst with temporal and channel encoding
    elif config.model_name == "patch_tst_tc":
        return PatchTransformerTC(
            c_in=config.feat_dim,
            c_out=c_out,
            num_patch=num_patch,
            patch_len=config.patch_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            shared_embedding=config.shared_embedding,
            norm=config.norm,
            activation=config.activation,
            pe="sincos",
            learn_pe=config.learn_pe,
            cls_token=config.cls_token,
            task=config.task,
            head_dropout=config.head_dropout,
        )
    # masked autoencoder with channel independence
    elif config.model_name == "masked_autoencoder":
        if config.task == "pretraining":
            return MaskedAutoencoder(
                c_in=config.feat_dim,
                num_patch=num_patch,
                patch_len=config.patch_len,
                enc_num_layers=config.enc_num_layers,
                enc_num_heads=config.enc_num_heads,
                enc_d_model=config.enc_d_model,
                enc_d_ff=config.enc_d_ff,
                dec_num_layers=config.dec_num_layers,
                dec_num_heads=config.dec_num_heads,
                dec_d_model=config.dec_d_model,
                dec_d_ff=config.dec_d_ff,
                dropout=config.dropout,
                shared_embedding=config.shared_embedding,
                norm=config.norm,
                activation=config.activation,
                pe="sincos",
                learn_pe=config.learn_pe,
                cls_token=config.cls_token,
                ch_token=config.ch_token,
            )
        elif config.task in ["classification", "forecasting"]:
            return MaskedAutoencoderPredictor(
                c_in=config.feat_dim,
                c_out=config.num_classes,
                num_patch=num_patch,
                patch_len=config.patch_len,
                enc_num_layers=config.enc_num_layers,
                enc_num_heads=config.enc_num_heads,
                enc_d_model=config.enc_d_model,
                enc_d_ff=config.enc_d_ff,
                dropout=config.dropout,
                shared_embedding=config.shared_embedding,
                norm=config.norm,
                activation=config.activation,
                pe="sincos",
                learn_pe=config.learn_pe,
                cls_token=config.cls_token,
                ch_token=config.ch_token,
                task=config.task,
                head_dropout=config.head_dropout,
            )
    # masked autoencoder with temporal encoding
    elif config.model_name == "masked_autoencoder_t":
        if config.task == "pretraining":
            return MaskedAutoencoderT(
                c_in=config.feat_dim,
                num_patch=num_patch,
                patch_len=config.patch_len,
                enc_num_layers=config.enc_num_layers,
                enc_num_heads=config.enc_num_heads,
                enc_d_model=config.enc_d_model,
                enc_d_ff=config.enc_d_ff,
                dec_num_layers=config.dec_num_layers,
                dec_num_heads=config.dec_num_heads,
                dec_d_model=config.dec_d_model,
                dec_d_ff=config.dec_d_ff,
                dropout=config.dropout,
                shared_embedding=config.shared_embedding,
                norm=config.norm,
                activation=config.activation,
                pe="sincos",
                learn_pe=config.learn_pe,
                cls_token=config.cls_token,
            )
        elif config.task in ["classification", "forecasting"]:
            return MaskedAutoencoderTPredictor(
                c_in=config.feat_dim,
                c_out=c_out,
                num_patch=num_patch,
                patch_len=config.patch_len,
                enc_num_layers=config.enc_num_layers,
                enc_num_heads=config.enc_num_heads,
                enc_d_model=config.enc_d_model,
                enc_d_ff=config.enc_d_ff,
                dropout=config.dropout,
                shared_embedding=config.shared_embedding,
                norm=config.norm,
                activation=config.activation,
                pe="sincos",
                learn_pe=config.learn_pe,
                cls_token=config.cls_token,
                task=config.task,
                head_dropout=config.head_dropout,
            )
    # masked autoencoder with temporal and channel encoding
    elif config.model_name == "masked_autoencoder_tc":
        if config.task == "pretraining":
            return MaskedAutoencoderTC(
                c_in=config.feat_dim,
                num_patch=num_patch,
                patch_len=config.patch_len,
                enc_num_layers=config.enc_num_layers,
                enc_num_heads=config.enc_num_heads,
                enc_d_model=config.enc_d_model,
                enc_d_ff=config.enc_d_ff,
                dec_num_layers=config.dec_num_layers,
                dec_num_heads=config.dec_num_heads,
                dec_d_model=config.dec_d_model,
                dec_d_ff=config.dec_d_ff,
                dropout=config.dropout,
                shared_embedding=config.shared_embedding,
                norm=config.norm,
                activation=config.activation,
                pe="sincos",
                learn_pe=config.learn_pe,
                cls_token=config.cls_token,
            )
        elif config.task in ["classification", "forecasting"]:
            return MaskedAutoencoderTCPredictor(
                c_in=config.feat_dim,
                c_out=c_out,
                num_patch=num_patch,
                patch_len=config.patch_len,
                enc_num_layers=config.enc_num_layers,
                enc_num_heads=config.enc_num_heads,
                enc_d_model=config.enc_d_model,
                enc_d_ff=config.enc_d_ff,
                dropout=config.dropout,
                shared_embedding=config.shared_embedding,
                norm=config.norm,
                activation=config.activation,
                pe="sincos",
                learn_pe=config.learn_pe,
                cls_token=config.cls_token,
                task=config.task,
                head_dropout=config.head_dropout,
            )
    else:
        raise ValueError(
            "Model class for task '{}' does not exist".format(config["task"])
        )


def optimizer_factory(config, model):
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError("Optimizer not specified")

    return optimizer


def scheduler_factory(config, optimizer, iters_per_epoch):
    if config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10, gamma=0.1
        )
    elif config.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=10
        )
    elif config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.epochs * iters_per_epoch
        )
    else:
        scheduler = None

    return scheduler


def tune_factory(config):
    if config.model_name == "patch_tst":
        config = get_patch_tst_search_space(config)
    if config.model_name == "pretraining_masked_autoencoder":
        config = get_pretraining_masked_autoencoder_search_space(config)
    return config
