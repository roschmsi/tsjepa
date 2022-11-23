from functools import partial
import torch
from data.dataset import (
    ClassificationDataset,
    ImputationDataset,
    collate_superv,
    collate_unsuperv,
)
from models.supervised_fedformer.model import (
    CNNFEDformerEncoder,
    CNNTimeFreqEncoder,
    CNNTimeFreqEncoderDecomp,
    FEDformer,
    FEDformerEncoder,
    FEDformerEncoderDecomp,
)
from models.supervised_cnn_transformer.model import CTN
from models.supervised_cnn_transformer.optimizer import NoamOpt
from models.unsupervised_transformer.optimizer import get_optimizer
from models.unsupervised_transformer.model import (
    TSTransformerEncoder,
    TSTransformerEncoderClassifier,
)
from running import SupervisedRunner, UnsupervisedRunner


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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    return optimizer

    if config.model.name == "unsupervised_transformer":
        if config.model["global_reg"]:
            weight_decay = config["l2_reg"]
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config.model["l2_reg"]

        optim_class = get_optimizer(config["optimizer"])
        optimizer = optim_class(
            model.parameters(), lr=config.training["lr"], weight_decay=weight_decay
        )
        return optimizer

    elif config.model.name == "supervised_cnn_transformer":
        optimizer = NoamOpt(
            model_size=config.model.d_model,
            factor=1,
            warmup=4000,
            optimizer=torch.optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
            ),
        )
        return optimizer

    elif (
        config.model.name == "supervised_fedformer"
        or config.model.name == "fedformer_encoder"
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
        return optimizer

    else:
        raise ValueError("No optimizer specified for this configuration.")


def model_factory(config):
    feat_dim = config.data.feat_dim  # dimensionality of data features
    max_seq_len = config.data.window * config.data.fs

    if config["task"] == "imputation":
        if config.model.name == "unsupervised_transformer_pretraining":
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

    elif config["task"] == "classification":
        if (
            config.model.name == "supervised_transformer"
            or config.model.name == "unsupervised_transformer_finetuning"
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
        elif config.model.name == "supervised_cnn_transformer":
            return CTN(
                feat_dim=feat_dim,
                d_model=config.model.d_model,
                num_heads=config.model.num_heads,
                d_ff=config.model.d_ff,
                num_layers=config.model.num_layers,
                num_classes=config.data.num_classes,
                max_seq_len=max_seq_len,
            )
        elif config.model.name == "fedformer_encoder":
            return FEDformerEncoder(config.model, config.data)
        elif config.model.name == "fedformer_cnn_encoder":
            return CNNFEDformerEncoder(config.model, config.data)
        elif config.model.name == "cnn_time_freq_encoder":
            return CNNTimeFreqEncoder(config.model, config.data)
        elif config.model.name == "fedformer_encoder_decomp":
            return FEDformerEncoderDecomp(config.model, config.data)
        elif config.model.name == "cnn_time_freq_encoder_decomp":
            return CNNTimeFreqEncoderDecomp(config.model, config.data)
        else:
            raise ValueError(
                "Model class for task '{}' does not exist".format(config["task"])
            )

    else:
        raise ValueError(
            "Model class for task '{}' does not exist".format(config["task"])
        )
