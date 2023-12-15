# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time
from functools import partial

import numpy as np
from utils.logging import log_training, readable_time
from model.forecaster import Forecaster
from runner.forecasting import ForecastingRunner
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import SupervisedDataset, create_patch, load_dataset
from data.ecg_dataset import classes, normal_class
from evaluation.evaluate_12ECG_score import compute_challenge_metric, load_weights
from model.revin import RevIN
from model.encoder import TransformerEncoder
from model.classifier import Classifier
from utils.helper import (
    load_encoder_from_tsjepa,
)
from model.setup import init_optimizer, init_scheduler
from utils.options import Options
from runner.classification import ClassificationRunner
from utils.helper import load_checkpoint, save_checkpoint
from utils.setup import seed_everything, setup

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
import pandas as pd

logger = logging.getLogger(__name__)
logger.info("Loading packages ...")


def main(config):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_epoch_time = 0
    total_start_time = time.time()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config.output_dir, "output.log"))
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
    dataset_class = SupervisedDataset

    # prepare dataloader
    train_dataset = dataset_class(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False if config.debug else True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_dataset = dataset_class(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_dataset = dataset_class(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # create Transformer encoder
    encoder = TransformerEncoder(
        patch_size=config.patch_len,
        num_patch=num_patch,
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

    # create model for downstream task
    if config.task == "forecasting":
        model = Forecaster(
            encoder=encoder,
            d_model=config.enc_d_model,
            num_patch=num_patch,
            forecast_len=config.pred_len,
            head_dropout=config.head_dropout,
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

    # create optimizer and scheduler
    optimizer = init_optimizer(
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = init_scheduler(optimizer, config)

    start_epoch = 0

    # load pretrained weights
    if config.load_model:
        if config.checkpoint_last:
            path = os.path.join(config.load_model, "checkpoints", "model_last.pth")
        elif config.checkpoint is not None:
            path = os.path.join(
                config.load_model, "checkpoints", f"model_{config.checkpoint}.pth"
            )
        else:
            path = os.path.join(config.load_model, "checkpoints", "model_best.pth")

        encoder = load_encoder_from_tsjepa(path=path, encoder=model.encoder)
        encoder = encoder.to(device)

        model.encoder = encoder

    # for linear probing
    if config.freeze:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False

    # choose task-dependent criterion
    if config.task == "forecasting":
        criterion = nn.MSELoss(reduction="mean")
    elif config.task == "classification":
        if config.multilabel:
            criterion = BCEWithLogitsLoss(reduction="mean")
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean")
    else:
        raise ValueError(f"Task {config.task} not supported.")

    # enable reversible instance normalization
    if config.revin:
        revin = RevIN(
            num_features=1,  # channel independence
            affine=config.revin_affine,
            subtract_last=False,
        )
        revin = revin.to(device)
    else:
        revin = None

    # setup trainer and evaluator
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

    trainer = runner_class(
        dataloader=train_loader, optimizer=optimizer, scheduler=scheduler
    )
    val_evaluator = runner_class(dataloader=val_loader)
    test_evaluator = runner_class(dataloader=test_loader)

    # only testing
    if config.test:
        logger.info("Test performance:")
        aggr_metrics_test, _ = test_evaluator.evaluate()
        for k, v in aggr_metrics_test.items():
            logger.info(f"{k}: {v}")

        return

    # robustness analysis with perturbations
    elif config.robustness:
        df = pd.DataFrame()

        with torch.no_grad():
            for i, perturbation_std in enumerate(
                np.arange(start=0, stop=1.1, step=0.1)
            ):
                logger.info("Perturbation std: ", perturbation_std)
                aggr_metrics_test = test_evaluator.evaluate(
                    perturbation_std=perturbation_std
                )

                logger.info(
                    f"Test performance with perturbation std {perturbation_std}:"
                )
                for k, v in aggr_metrics_test.items():
                    logger.info(f"{k}: {v}")

                aggr_metrics_test["perturbation_std"] = perturbation_std
                df = pd.concat([df, pd.DataFrame(aggr_metrics_test, index=[i])])

        df.to_csv(os.path.join(config.output_dir, "robustness.csv"))
        return

    tb_writer = SummaryWriter(config.output_dir)

    patience_count = 0
    best_loss_val = 1e16

    logger.info("Starting training...")

    for epoch in tqdm(
        range(start_epoch + 1, config.epochs + 1),
        desc="Training Epoch",
        leave=False,
    ):
        # option to unfreeze entire model after initial linear probing
        if "freeze" in config.keys():
            if config.freeze and epoch > config.freeze_epochs:
                for name, param in model.named_parameters():
                    param.requires_grad = True

        # train model
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch()
        epoch_end_time = time.time()

        log_training(
            epoch=epoch,
            aggr_metrics_train=aggr_metrics_train,
            tb_writer=tb_writer,
            start_epoch=start_epoch,
            total_epoch_time=total_epoch_time,
            epoch_start_time=epoch_start_time,
            epoch_end_time=epoch_end_time,
            num_batches=len(train_loader),
            num_samples=len(train_dataset),
        )

        # evaluate model
        if epoch % config.val_interval == 0:
            with torch.no_grad():
                aggr_metrics_val = val_evaluator.evaluate(epoch)

            for k, v in aggr_metrics_val.items():
                tb_writer.add_scalar(f"{k}/val", v, epoch)

            if aggr_metrics_val["loss"] < best_loss_val:
                best_loss_val = aggr_metrics_val["loss"]
                patience_count = 0
                better = True
            else:
                patience_count += config.val_interval
                better = False

            save_checkpoint(
                epoch=epoch,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                path=config.checkpoint_dir,
                better=better,
            )

        if patience_count > config.patience:
            break

    # load best model
    path = os.path.join(config.output_dir, "checkpoints", "model_best.pth")
    model, _, _, epoch = load_checkpoint(
        path,
        model=model,
    )

    # evaluate on test set
    with torch.no_grad():
        test_evaluator.model = model
        aggr_metrics_test = test_evaluator.evaluate()

    logger.info("Best test performance:")
    for k, v in aggr_metrics_test.items():
        logger.info(f"{k}: {v}")

    # compute physionet challenge metric
    if config.dataset == "ecg" and config.task == "classification":
        logger.info("Compute PhysioNet 2020 challenge metric")
        step = 0.02
        scores = []
        weights = load_weights(config.weights_file, classes)

        for thr in np.arange(0.0, 1.0, step):
            lbls = []
            probs = []

            for batch in val_loader:
                X, targets = batch
                X = X.to(device)
                X = create_patch(X, patch_len=config.patch_len, stride=config.stride)
                targets = targets.to(device)

                with torch.no_grad():
                    predictions = model(X)
                    prob = predictions.sigmoid().cpu().numpy()
                    probs.append(prob)
                    lbls.append(targets.cpu().numpy())

            lbls = np.concatenate(lbls)
            probs = np.concatenate(probs)

            preds = (probs > thr).astype(np.int)
            challenge_metric = compute_challenge_metric(
                weights, lbls, preds, classes, normal_class
            )
            scores.append(challenge_metric)

        # best thrs and preds
        scores = np.array(scores)
        idxs = np.argmax(scores, axis=0)
        thr_final = idxs * step

        logger.info(f"Validation challenge score: {scores[idxs]}")
        logger.info(f"Threshold: {thr_final}")

        # physionet 2020 challenge
        lbls = []
        probs = []

        for batch in test_loader:
            X, targets = batch
            X = X.to(device)
            X = create_patch(X, patch_len=config.patch_len, stride=config.stride)
            targets = targets.to(device)

            with torch.no_grad():
                predictions = model(X)
                prob = predictions.sigmoid().cpu().numpy()
                probs.append(prob)
                lbls.append(targets.cpu().numpy())

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)

        preds = (probs > thr_final).astype(np.int)
        challenge_metric_test = compute_challenge_metric(
            weights, lbls, preds, classes, normal_class
        )

        logger.info(f"Test challenge score: {challenge_metric_test}")

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
