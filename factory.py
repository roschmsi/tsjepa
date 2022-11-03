from functools import partial
import torch
from data.dataset import (
    ClassificationDataset,
    ImputationDataset,
    collate_superv,
    collate_unsuperv,
)
from models.supervised_fedformer.model import FEDformer
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
    if config.model.name == "unsupervised_transformer":
        # initialize optimizer and regularization
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
        # optimizer = NoamOpt(
        #     model_size=config.model.d_model,
        #     factor=1,
        #     warmup=4000,
        #     optimizer=torch.optim.Adam(
        #         model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        #     ),
        # )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
        return optimizer

    elif config.model.name == "supervised_fedformer":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
        return optimizer

    else:
        raise ValueError("No optimizer specified for this configuration.")


def model_factory(config):
    feat_dim = config.data.feat_dim  # dimensionality of data features
    max_seq_len = config.data.window * config.data.fs

    if config["task"] == "imputation":
        if config.model.name == "unsupervised_transformer":
            return TSTransformerEncoder(
                feat_dim,
                max_seq_len,
                config.model["d_model"],
                config.model["num_heads"],
                config.model["num_layers"],
                config.model["dim_feedforward"],
                dropout=config.model["dropout"],
                pos_encoding=config.model["pos_encoding"],
                activation=config.model["activation"],
                norm=config.model["normalization_layer"],
                freeze=config.model["freeze"],
            )

    elif config["task"] == "classification":
        if config.model.name == "unsupervised_transformer":
            return TSTransformerEncoderClassifier(
                feat_dim,
                max_seq_len,
                config.model["d_model"],
                config.model["num_heads"],
                config.model["num_layers"],
                config.model["dim_feedforward"],
                num_classes=config.data.num_classes,
                dropout=config.model["dropout"],
                pos_encoding=config.model["pos_encoding"],
                activation=config.model["activation"],
                norm=config.model["normalization_layer"],
                freeze=config.model["freeze"],
            )
        elif config.model.name == "supervised_cnn_transformer":
            return CTN(
                d_model=config.model.d_model,
                nhead=config.model.nhead,
                d_ff=config.model.d_ff,
                num_layers=config.model.num_layers,
                num_classes=config.data.num_classes,
            )
        elif config.model.name == "supervised_fedformer":
            return FEDformer(config.model, config.data)
        else:
            raise ValueError(
                "Model class for task '{}' does not exist".format(config["task"])
            )

    else:
        raise ValueError(
            "Model class for task '{}' does not exist".format(config["task"])
        )
