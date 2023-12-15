# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time
from functools import partial
from model.forecaster import Forecaster
from runner.forecasting import ForecastingRunner

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import load_dataset
from model.setup import (
    init_optimizer,
    init_scheduler,
)
from options import Options
from runner.classification import ClassificationRunner
from utils import load_checkpoint, save_checkpoint
from utils_old import log_training, readable_time, seed_everything, setup
from model.revin import RevIN

from model.classifier import Classifier
from model.encoder import TransformerEncoder
from utils import (
    load_encoder_from_tsjepa,
)
from data.dataset import SupervisedDataset
import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")


def main(config):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_epoch_time = 0
    total_start_time = time.time()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))

    # config for debugging on single sample
    if config.debug:
        config.batch_size = 2
        config.val_interval = 10
        config.plot_interval = 10
        config.augment = False
        config.dropout = 0

    # build data
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    if config.seq_len is not None:
        max_seq_len = config.seq_len
    else:
        max_seq_len = config.window * config.fs

    if config.use_patch:
        num_patch = (max_seq_len - config.patch_len) // config.stride + 1

    # initialize data generator and runner
    # channel independence, TODO solve for forecasting all dataset
    dataset_class = SupervisedDataset

    test_dataset = dataset_class(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        # collate_fn=mask_collator,
    )

    # create model
    encoder = TransformerEncoder(
        seq_len=max_seq_len,
        patch_size=config.patch_len,
        num_patch=num_patch,
        in_chans=config.feat_dim,
        embed_dim=config.enc_d_model,
        depth=config.enc_num_layers,
        num_heads=config.enc_num_heads,
        mlp_ratio=config.enc_mlp_ratio,
        drop_rate=config.dropout,
        attn_drop_rate=config.attn_drop_rate,
        activation=config.activation,
        activation_drop_rate=config.activation_drop_rate,
        norm=config.norm,
        layer_norm_first=config.layer_norm_first,
        learn_pe=config.learn_pe,
    )
    if config.task == "forecasting":
        model = Forecaster(
            encoder=encoder,
            n_vars=config.feat_dim,
            d_model=config.enc_d_model,
            num_patch=num_patch,
            forecast_len=config.pred_len,
            patch_len=config.patch_len,
            head_dropout=config.head_dropout,
            head_type=config.head_type,
        )
    elif config.task == "classification":
        model = Classifier(
            encoder=encoder,
            n_vars=config.feat_dim,
            d_model=config.enc_d_model,
            n_classes=config.num_classes,
            head_dropout=config.head_dropout,
        )
    else:
        raise ValueError(f"Task {config.task} not supported.")

    model = model.to(device)

    # TODO ensure that model is loaded
    if config.checkpoint is not None:
        path = os.path.join(
            config.load_model, "checkpoints", f"model_{config.checkpoint}.pth"
        )
    else:
        path = os.path.join(config.load_model, "checkpoints", "model_best.pth")

    model, _, _, _ = load_checkpoint(
        path=path, model=model, optimizer=None, scheduler=None
    )

    # TODO enable to select a loss
    if config.task == "forecasting":
        criterion = nn.MSELoss(reduction="mean")
    elif config.task == "classification":
        if config.multilabel:
            criterion = BCEWithLogitsLoss(reduction="mean")
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean")
    else:
        raise ValueError(f"Task {config.task} not supported.")

    if config.revin:
        revin = RevIN(
            num_features=1,  # config.feat_dim,
            affine=config.revin_affine,
            subtract_last=False,
        )
        revin = revin.to(device)
    else:
        revin = None

    if config.task == "forecasting":
        runner_class = ForecastingRunner
        runner_class = partial(
            runner_class,
            model=model,
            revin=revin,
            device=device,
            criterion=criterion,
            patch_len=config.patch_len,
            stride=config.stride,
        )
    elif config.task == "classification":
        runner_class = ClassificationRunner
        runner_class = partial(
            runner_class,
            model=model,
            device=device,
            criterion=criterion,
            patch_len=config.patch_len,
            stride=config.stride,
        )
    else:
        raise ValueError(f"Task {config.task} not supported.")

    test_evaluator = runner_class(dataloader=test_loader)

    # save results in a dataframe
    # tb_writer = SummaryWriter(config.output_dir)
    df = pd.DataFrame()

    # TODO load best model, evaluate on test set
    with torch.no_grad():
        for i, perturbation_std in enumerate(np.arange(start=0, stop=1.1, step=0.1)):
            print("perturbation std: ", perturbation_std)
            aggr_metrics_test = test_evaluator.evaluate(
                perturbation_std=perturbation_std
            )

            logger.info(f"Test performance with perturbation std {perturbation_std}:")
            for k, v in aggr_metrics_test.items():
                logger.info(f"{k}: {v}")

            aggr_metrics_test["perturbation_std"] = perturbation_std
            df = pd.concat([df, pd.DataFrame(aggr_metrics_test, index=[i])])

    df.to_csv(os.path.join(config.output_dir, "robustness.csv"))

    logger.info("Test performance:")
    for k, v in aggr_metrics_test.items():
        logger.info(f"{k}: {v}")

    total_runtime = time.time() - total_start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(total_runtime)
        )
    )
    logger.info("Done.")


if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    main(config)
