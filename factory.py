from functools import partial

import torch

from data.dataset import (
    ClassificationDataset,
    ClassificationPatchDataset,
    ImputationDataset,
    ImputationPatchDataset,
    collate_patch_superv,
    collate_patch_unsuperv,
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
from models.gtn.model import GatedTransformer
from models.residual_cnn_att.model import ResidualCNNAtt
from models.transformer.model import (
    TSTransformerEncoder,
    TSTransformerEncoderClassifier,
)
from models.transformer.optimizer import RAdam
from runner import SupervisedRunner, UnsupervisedPatchRunner, UnsupervisedRunner

from models.patch_tst.model import PatchTST


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    if config["task"] == "imputation" and not config.model.use_patch:
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
    elif config["task"] == "imputation" and config.model.use_patch:
        return (
            partial(
                ImputationPatchDataset,
                masking_ratio=config.model["masking_ratio"],
                patch_len=config.model.patch_len,
                stride=config.model.stride,
            ),
            collate_patch_unsuperv,
            UnsupervisedPatchRunner,
        )
    elif config["task"] == "classification" and not config.model.use_patch:
        return ClassificationDataset, collate_superv, SupervisedRunner
    elif config["task"] == "classification" and config.model.use_patch:
        return (
            partial(
                ClassificationPatchDataset,
                patch_len=config.model.patch_len,
                stride=config.model.stride,
            ),
            collate_patch_superv,
            SupervisedRunner,
        )
    else:
        raise NotImplementedError("Task '{}' not implemented".format(config["task"]))


def optimizer_factory(config, model):
    # for initial experiments only use Adam optimizer
    if config.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "RAdam":
        optimizer = RAdam(model.parameters(), lr=config.training.lr)
    return optimizer


def model_factory(config):
    if "max_seq_len" in config.data.keys():
        max_seq_len = config.data.max_seq_len
    else:
        max_seq_len = config.data.window * config.data.fs

    if config.model.name == "pretraining_transformer":
        return TSTransformerEncoder(
            feat_dim=config.data.feat_dim,
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
            feat_dim=config.data.feat_dim,
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
            feat_dim=config.data.feat_dim,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            num_layers=config.model.num_layers,
            num_classes=config.data.num_classes,
            max_seq_len=max_seq_len,
            dropout=config.model.dropout,
            num_cnn=config.model.num_cnn,
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
            feat_dim=config.data.feat_dim,
            d_model=config.model.d_model,
            num_classes=config.data.num_classes,
            num_cnn=config.model.num_cnn,
        )
    elif config.model.name == "cnn_encoder_3l":
        return CNNEncoder3L(
            feat_dim=config.data.feat_dim,
            d_model=config.model.d_model,
            num_classes=config.data.num_classes,
        )
    elif config.model.name == "residual_cnn_att":
        return ResidualCNNAtt(nOUT=config.data.num_classes)
    elif config.model.name == "gtn":
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return GatedTransformer(
            d_model=config.model.d_model,
            d_input=max_seq_len,
            d_channel=config.data.feat_dim,
            d_output=config.data.num_classes,
            d_hidden=config.model.d_ff,
            q=config.model.q,
            v=config.model.v,
            h=config.model.h,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            pe=True,
            mask=config.model.mask,
            device=DEVICE,
        )
    elif config.model.name == "finetuning_patch_tst":
        num_patch = (
            max(max_seq_len, config.model.patch_len) - config.model.patch_len
        ) // config.model.stride + 1
        return PatchTST(
            c_in=config.data.feat_dim,
            target_dim=config.data.num_classes,
            patch_len=config.model.patch_len,
            stride=config.model.stride,
            num_patch=num_patch,
            n_layers=config.model.num_layers,
            n_heads=config.model.num_heads,
            d_model=config.model.d_model,
            shared_embedding=True,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            head_dropout=config.model.head_dropout,
            act="relu",
            head_type="classification",
            res_attention=False,
        )
    elif config.model.name == "pretraining_patch_tst":
        num_patch = (
            max(max_seq_len, config.model.patch_len) - config.model.patch_len
        ) // config.model.stride + 1
        print("number of patches:", num_patch)
        return PatchTST(
            c_in=config.data.feat_dim,
            target_dim=config.data.feat_dim,
            patch_len=config.model.patch_len,
            stride=config.model.stride,
            num_patch=num_patch,
            n_layers=config.model.num_layers,
            n_heads=config.model.num_heads,
            d_model=config.model.d_model,
            shared_embedding=True,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            head_dropout=config.model.head_dropout,
            act="relu",
            head_type="pretrain",
            res_attention=False,
        )
    elif config.model.name == "patch_tst":
        num_patch = (
            max(max_seq_len, config.model.patch_len) - config.model.patch_len
        ) // config.model.stride + 1
        print("number of patches:", num_patch)

        return PatchTST(
            c_in=config.data.feat_dim,
            target_dim=config.data.num_classes,
            patch_len=config.model.patch_len,
            stride=config.model.stride,
            num_patch=num_patch,
            n_layers=config.model.num_layers,
            n_heads=config.model.num_heads,
            d_model=config.model.d_model,
            shared_embedding=True,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            head_dropout=config.model.head_dropout,
            act="relu",
            head_type="classification",
            res_attention=False,
        )
    else:
        raise ValueError(
            "Model class for task '{}' does not exist".format(config["task"])
        )
