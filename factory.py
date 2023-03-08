from functools import partial

import torch

from data.dataset import (
    ClassificationDataset,
    MAEClassificationPatchDataset,
    ClassificationPatchDataset,
    PretrainingDataset,
    MAEPretrainingPatchDataset,
    PretrainingPatchDataset,
    collate_patch_superv,
    collate_patch_unsuperv,
    collate_superv,
    collate_unsuperv,
)
from models.cnn_transformer.model import (
    CNNEncoder,
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
from models.masked_autoencoder.model import (
    MaskedAutoencoderTST,
    MaskedAutoencoderTSTClassifier,
)
from models.masked_autoencoder.pretraining_masked_autoencoder_search_space import (
    get_pretraining_masked_autoencoder_search_space,
)
from models.patch_tst.patch_tst_search_space import (
    get_patch_tst_search_space,
)
from models.patch_tst_2d.model import PatchTST2d
from models.residual_cnn_att.model import ResidualCNNAtt
from models.transformer.model import (
    TSTransformerEncoder,
    TSTransformerEncoderClassifier,
)
from models.transformer.optimizer import RAdam
from runner import (
    SupervisedRunner,
    UnsupervisedMAERunner,
    UnsupervisedPatchRunner,
    UnsupervisedRunner,
)

from models.patch_tst.model import PatchTST


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    if config.model_name == "pretraining_transformer":
        return (
            partial(
                PretrainingDataset,
                mean_mask_length=config.mean_mask_length,
                masking_ratio=config.masking_ratio,
                mode=config.mask_mode,
                distribution=config.mask_distribution,
                exclude_feats=config.exclude_feats,
            ),
            collate_unsuperv,
            UnsupervisedRunner,
        )
    elif config.model_name == "pretraining_masked_autoencoder":
        return (
            partial(
                MAEPretrainingPatchDataset,
                masking_ratio=config.masking_ratio,
                patch_len=config.patch_len,
                stride=config.stride,
                debug=config.debug,
            ),
            partial(
                collate_patch_unsuperv,
                patch_len=config.patch_len,
                stride=config.stride,
                masking_ratio=config.masking_ratio,
            ),
            UnsupervisedMAERunner,
        )
    elif config.model_name == "finetuning_masked_autoencoder":
        return (
            partial(
                MAEClassificationPatchDataset,
                masking_ratio=config.masking_ratio,
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
    elif config.model_name in ["pretraining_patch_tst", "pretraining_patch_tst_2d"]:
        return (
            partial(
                PretrainingPatchDataset,
                masking_ratio=config.masking_ratio,
                patch_len=config.patch_len,
                stride=config.stride,
                debug=config.debug,
            ),
            partial(
                collate_patch_unsuperv,
                patch_len=config.patch_len,
                stride=config.stride,
                masking_ratio=config.masking_ratio,
            ),
            UnsupervisedPatchRunner,
        )
    elif config.model_name == "finetuning_patch_tst":
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
    elif config["task"] == "classification" and not config.use_patch:
        return ClassificationDataset, collate_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(config["task"]))


def optimizer_factory(config, model):
    # for initial experiments only use Adam optimizer
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
    elif config.optimizer == "RAdam":
        optimizer = RAdam(model.parameters(), lr=config.lr)
    return optimizer


def scheduler_factory(config, optimizer):
    if config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10, gamma=0.1
        )
    elif config.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=10
        )
    else:
        scheduler = None

    return scheduler


def model_factory(config):
    if "max_seq_len" in config.keys():
        max_seq_len = config.max_seq_len
    else:
        max_seq_len = config.window * config.fs

    if config.use_patch:
        num_patch = (max_seq_len - config.patch_len) // config.stride + 1

    if config.model_name == "pretraining_transformer":
        return TSTransformerEncoder(
            feat_dim=config.feat_dim,
            max_len=max_seq_len,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            pos_encoding=config.pos_encoding,
            activation=config.activation,
            norm=config.normalization_layer,
            freeze=config.freeze,
        )
    elif (
        config.model_name == "transformer"
        or config.model_name == "finetuning_transformer"
    ):
        return TSTransformerEncoderClassifier(
            feat_dim=config.feat_dim,
            max_len=max_seq_len,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            pos_encoding=config.pos_encoding,
            activation=config.activation,
            norm=config.normalization_layer,
            num_classes=config.num_classes,
            freeze=config.freeze,
        )
    elif config.model_name == "cnn_transformer":
        return CNNTransformer(
            feat_dim=config.feat_dim,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            max_seq_len=max_seq_len,
            dropout=config.dropout,
            num_cnn=config.num_cnn,
        )
    elif config.model_name == "fedformer_encoder":
        return FEDformerEncoder(config.model, config.data)
    elif config.model_name == "cnn_fedformer_encoder":
        return CNNFEDformerEncoder(config.model, config.data)
    elif config.model_name == "decomp_fedformer_encoder":
        return DecompFEDformerEncoder(config.model, config.data)
    elif config.model_name == "cnn_time_freq_encoder":
        return CNNTimeFreqEncoder(config.model, config.data)
    elif config.model_name == "cnn_decomp_time_freq_encoder":
        return CNNDecompTimeFreqEncoder(config.model, config.data)
    elif config.model_name == "cnn_encoder":
        return CNNEncoder(
            feat_dim=config.feat_dim,
            d_model=config.d_model,
            num_classes=config.num_classes,
            num_cnn=config.num_cnn,
        )
    elif config.model_name == "residual_cnn_att":
        return ResidualCNNAtt(nOUT=config.num_classes)
    elif config.model_name == "gtn":
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return GatedTransformer(
            d_model=config.d_model,
            d_input=max_seq_len,
            d_channel=config.feat_dim,
            d_output=config.num_classes,
            d_hidden=config.d_ff,
            q=config.q,
            v=config.v,
            h=config.h,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pe=True,
            mask=config.mask,
            device=DEVICE,
        )
    elif config.model_name == "pretraining_patch_tst":
        return PatchTST(
            c_in=config.feat_dim,
            target_dim=config.feat_dim,
            patch_len=config.patch_len,
            stride=config.stride,
            num_patch=num_patch,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            shared_embedding=True,
            d_ff=config.d_ff,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            act="relu",
            head_type="pretrain",
            res_attention=False,
        )
    elif config.model_name == "pretraining_patch_tst_2d":
        return PatchTST2d(
            c_in=config.feat_dim,
            target_dim=config.feat_dim,
            patch_len=config.patch_len,
            stride=config.stride,
            num_patch=num_patch,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            shared_embedding=True,
            d_ff=config.d_ff,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            act="relu",
            head_type="pretrain",
            res_attention=False,
        )
    elif config.model_name == "finetuning_patch_tst":
        return PatchTST(
            c_in=config.feat_dim,
            target_dim=config.num_classes,
            patch_len=config.patch_len,
            stride=config.stride,
            num_patch=num_patch,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            shared_embedding=True,
            d_ff=config.d_ff,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            act="relu",
            head_type="classification",
            res_attention=False,
        )
    elif config.model_name == "patch_tst":
        return PatchTST(
            c_in=config.feat_dim,
            target_dim=config.num_classes,
            patch_len=config.patch_len,
            stride=config.stride,
            num_patch=num_patch,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            shared_embedding=True,
            d_ff=config.d_ff,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            act="relu",
            head_type="classification",
            res_attention=False,
        )
    elif config.model_name == "pretraining_masked_autoencoder":
        return MaskedAutoencoderTST(
            num_patch=num_patch,
            patch_len=config.patch_len,
            masking_ratio=config.masking_ratio,
            c_in=config.feat_dim,
            enc_num_layers=config.enc_num_layers,
            enc_num_heads=config.enc_num_heads,
            enc_d_model=config.enc_d_model,
            enc_d_ff=config.enc_d_ff,
            dec_num_layers=config.dec_num_layers,
            dec_num_heads=config.dec_num_heads,
            dec_d_model=config.dec_d_model,
            dec_d_ff=config.dec_d_ff,
            shared_embedding=True,
            dropout=config.dropout,
            act="relu",
            res_attention=False,
            pe="sincos",
        )
    elif config.model_name == "finetuning_masked_autoencoder":
        return MaskedAutoencoderTSTClassifier(
            num_patch=num_patch,
            patch_len=config.patch_len,
            masking_ratio=config.masking_ratio,
            c_in=config.feat_dim,
            target_dim=config.num_classes,
            enc_d_model=config.enc_d_model,
            enc_d_ff=config.enc_d_ff,
            enc_num_layers=config.enc_num_layers,
            enc_num_heads=config.enc_num_heads,
            shared_embedding=True,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            act="relu",
            res_attention=False,
            pe="sincos",
        )
    else:
        raise ValueError(
            "Model class for task '{}' does not exist".format(config["task"])
        )


def tune_factory(config):
    if config.model_name == "patch_tst":
        config = get_patch_tst_search_space(config)
    if config.model_name == "pretraining_masked_autoencoder":
        config = get_pretraining_masked_autoencoder_search_space(config)
    return config
