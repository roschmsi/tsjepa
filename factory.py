from functools import partial

import torch

from data.dataset import (
    ClassificationDataset,
    ImputationDataset,
    collate_superv,
    collate_unsuperv,
)
from models.cnn_transformer.model import (
    CNNEncoder,
    CNNEncoder3L,
    CNNTransformer,
)
from models.fedformer.model import (
    CNNDecompTimeFreqEncoder,
    CNNFEDformerEncoder,
    CNNTimeFreqEncoder,
    DecompFEDformerEncoder,
    FEDformerEncoder,
)
from models.transformer.model import (
    TSTransformerEncoder,
    TSTransformerEncoderClassifier,
)
from models.transformer.optimizer import RAdam
from runner import SupervisedRunner, UnsupervisedRunner


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    if config["task"] == "imputation":
        return (
            partial(
                ImputationDataset,
                mean_mask_length=config.model["mean_mask_length"],
                masking_ratio=config.model["masking_ratio"],
                mode=config.model["mask_mode"],
                distribution=config.model["mask_distribution"],
                exclude_feats=config.model["exclude_feats"],
            ),
            collate_unsuperv,
            UnsupervisedRunner,
        )
    elif config["task"] == "classification":
        return ClassificationDataset, collate_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(config["task"]))


def optimizer_factory(config, model):
    # for initial experiments only use Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        #  weight_decay=0.1
    )
    return optimizer


def model_factory(config):
    feat_dim = config.data.feat_dim  # dimensionality of data features
    if "max_seq_len" in config.data.keys():
        max_seq_len = config.data.max_seq_len
    else:
        max_seq_len = config.data.window * config.data.fs

    if config.model.name == "pretraining_transformer":
        return TSTransformerEncoder(
            feat_dim=feat_dim,
            max_len=max_seq_len,
            d_model=config.model["d_model"],
            num_heads=config.model["num_heads"],
            num_layers=config.model["num_layers"],
            d_ff=config.model["d_ff"],
            dropout=config.model["dropout"],
            pos_encoding=config.model["pos_encoding"],
            activation=config.model["activation"],
            norm=config.model["normalization_layer"],
            freeze=config.model["freeze"],
        )
    elif (
        config.model.name == "transformer"
        or config.model.name == "finetuning_transformer"
    ):
        return TSTransformerEncoderClassifier(
            feat_dim=feat_dim,
            max_len=max_seq_len,
            d_model=config.model["d_model"],
            num_heads=config.model["num_heads"],
            num_layers=config.model["num_layers"],
            d_ff=config.model["d_ff"],
            num_classes=config.data.num_classes,
            dropout=config.model["dropout"],
            pos_encoding=config.model["pos_encoding"],
            activation=config.model["activation"],
            norm=config.model["normalization_layer"],
            freeze=config.model["freeze"],
        )
    elif config.model.name == "cnn_transformer":
        return CNNTransformer(
            feat_dim=feat_dim,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            num_layers=config.model.num_layers,
            num_classes=config.data.num_classes,
            max_seq_len=max_seq_len,
            dropout=config.model.dropout,
        )
    elif config.model.name == "fedformer_encoder":
        return FEDformerEncoder(config.model, config.data)
    elif config.model.name == "cnn_fedformer_encoder":
        return CNNFEDformerEncoder(config.model, config.data)
    elif config.model.name == "decomp_fedformer_encoder":
        return DecompFEDformerEncoder(config.model, config.data)
    elif config.model.name == "cnn_time_freq_encoder":
        return CNNTimeFreqEncoder(config.model, config.data)
    elif config.model.name == "cnn_decomp_time_freq_encoder":
        return CNNDecompTimeFreqEncoder(config.model, config.data)
    elif config.model.name == "cnn_encoder":
        return CNNEncoder(
            feat_dim=feat_dim,
            d_model=config.model.d_model,
            num_classes=config.data.num_classes,
        )
    elif config.model.name == "cnn_encoder_3l":
        return CNNEncoder3L(
            feat_dim=feat_dim,
            d_model=config.model.d_model,
            num_classes=config.data.num_classes,
        )
    else:
        raise ValueError(
            "Model class for task '{}' does not exist".format(config["task"])
        )
