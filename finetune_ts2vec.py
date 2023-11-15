# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time
from functools import partial

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import load_dataset
from models.ts_jepa.setup import (
    init_optimizer,
    init_scheduler,
)
from options import Options
from runner.ts2vec import TS2VecForecastingRunner, TS2VecClassificationRunner
from utils import log_training, readable_time, seed_everything, setup
from models.patch_tst.layers.revin import RevIN

from models.ts2vec.ts2vec import TS2VecForecaster, TS2VecClassifier
from models.ts2vec.encoder import TransformerEncoder
from models.ts2vec.utils import (
    load_encoder_from_ts2vec,
    save_checkpoint,
    load_checkpoint,
)
from data.dataset import JEPADataset

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
    dataset_class = JEPADataset

    # prepare dataloader
    train_dataset = dataset_class(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False if config.debug else True,
        num_workers=config.num_workers,
        pin_memory=True,
        # collate_fn=mask_collator,
    )
    val_dataset = dataset_class(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        # collate_fn=mask_collator,
    )
    test_dataset = dataset_class(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        # collate_fn=mask_collator,
    )

    ipe = len(train_loader)

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
        cls_token=config.cls_token,
    )
    if config.task == "forecasting":
        model = TS2VecForecaster(
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
        model = TS2VecClassifier(
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
        epochs=config.epochs,
    )
    scheduler = init_scheduler(optimizer, config)

    start_epoch = 0

    if config.load_model:
        # load pretrained weights
        # TODO load checkpoint best or last or specified one with single string
        if config.checkpoint_last:
            path = os.path.join(config.load_model, "checkpoints", "model_last.pth")
        elif config.checkpoint is not None:
            path = os.path.join(
                config.load_model, "checkpoints", f"model_{config.checkpoint}.pth"
            )
        else:
            path = os.path.join(config.load_model, "checkpoints", "model_best.pth")

        encoder = load_encoder_from_ts2vec(path=path, encoder=model.encoder)
        encoder = encoder.to(device)

        model.encoder = encoder

    if config.freeze:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False

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
        runner_class = TS2VecForecastingRunner
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
        runner_class = TS2VecClassificationRunner
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

    if config.test:
        logger.info("Test performance:")
        aggr_metrics_test, _ = test_evaluator.evaluate()
        for k, v in aggr_metrics_test.items():
            logger.info(f"{k}: {v}")

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
        aggr_metrics_train = trainer.train_epoch(epoch)
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
        if epoch % config["val_interval"] == 0:
            with torch.no_grad():
                aggr_metrics_val = val_evaluator.evaluate(epoch)

            for k, v in aggr_metrics_val.items():
                tb_writer.add_scalar(f"{k}/val", v, epoch)

            if aggr_metrics_val["loss"] < best_loss_val:
                best_loss_val = aggr_metrics_val["loss"]
                patience_count = 0
                better = True
            else:
                patience_count += config["val_interval"]
                better = False

            save_checkpoint(
                epoch=epoch,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                path=config["checkpoint_dir"],
                better=better,
            )

        if patience_count > config.patience:
            break

    # load best model, iterate over test loader, get representations
    # mean over time axis, then plot with tsne

    # if not config.debug:
    path = os.path.join(config["output_dir"], "checkpoints", "model_best.pth")
    model, _, _, epoch = load_checkpoint(
        path,
        model=model,
    )

    # TODO load best model, evaluate on test set
    with torch.no_grad():
        test_evaluator.model = model
        aggr_metrics_test = test_evaluator.evaluate(epoch_num=epoch)

    logger.info("Best test performance:")
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
