# Reference: https://github.com/gzerveas/mvts_transformer

import copy
import logging
import os
import sys
import time
from functools import partial


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import JEPADataset, load_dataset, ConcatenatedDataset, CIDataset
from models.ts_jepa.mask import RandomMaskCollator, BlockMaskCollator
from models.ts_jepa.setup import (
    init_model_pretraining,
    init_optimizer_enc_pred,
    init_optimizer_model,
    init_scheduler,
)
from models.ts_jepa.utils import (
    plot_2d,
    plot_classwise_distribution,
)
from models.ts2vec.utils import save_checkpoint, load_checkpoint
from options import Options
from runner.ts2vec import TS2VecRunner
from utils import log_training, readable_time, seed_everything, setup
from models.patch_tst.layers.revin import RevIN

from models.ts2vec.ts2vec import TS2Vec
from models.ts2vec.encoder import TransformerEncoder

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")  #

from data.dataset import CIPretrainingPatchDataset, collate_patch_unsuperv


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

    # initialize data generator and runner
    # channel independence, TODO solve for forecasting all dataset
    dataset_class = partial(CIDataset, num_channels=config.feat_dim)

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
        in_chans=config.feat_dim,
        embed_dim=config.enc_d_model,
        depth=config.enc_num_layers,
        num_heads=config.enc_num_heads,
        mlp_ratio=config.enc_mlp_ratio,
        drop_rate=config.dropout,
        attn_drop_rate=config.attn_drop_rate,
        activation=config.activation,
        activation_drop_rate=config.activation_drop_rate,
        layer_norm_first=config.layer_norm_first,
        learn_pe=config.learn_pe,
    )
    model = TS2Vec(
        encoder=encoder,
        device=device,
        average_top_k_layers=config.average_top_k_layers,
        normalize_targets=config.normalize_targets,
        targets_norm=config.targets_norm,
        embed_dim=config.enc_d_model,
        ema_decay=config.ema_decay,
        ema_end_decay=config.ema_end_decay,
        ema_anneal_end_step=config.ema_anneal_end_step * ipe,
        skip_embeddings=config.skip_embeddings,
    )
    model.to(device)

    # create optimizer and scheduler
    # TODO use more advanced optimizer from setup.py
    optimizer = init_optimizer_model(
        model, lr=config.lr, weight_decay=config.weight_decay, epochs=config.epochs
    )

    scheduler = init_scheduler(optimizer, config)

    start_epoch = 0

    # load pretrained weights
    if config.resume or config.load_model:
        if config.resume:
            path = os.path.join(config["output_dir"], "checkpoints", "model_last.pth")
        elif config.load_model:
            path = os.path.join(config["output_dir"], "checkpoints", "model_best.pth")
        # TODO load checkpoint
        model, optimizer, epoch = load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
        )
        if config.resume:
            start_epoch = epoch
            if scheduler is not None:
                for _ in range(start_epoch):
                    scheduler.step()

    if config.revin:
        revin = RevIN(
            num_features=1,  # channel independence
            affine=True if config.revin_affine else False,
            subtract_last=False,
        )
        # revin = revin.to(device)
    else:
        revin = None

    # initialize runner for training, validation and testing
    runner_class = TS2VecRunner
    runner_class = partial(
        runner_class,
        model=model,
        revin=revin,
        device=device,
        patch_len=config.patch_len,
        stride=config.stride,
        masking=config.masking,
        masking_ratio=config.masking_ratio,
        debug=config.debug,
    )
    trainer = runner_class(
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
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
        # train model
        epoch_start_time = time.time()
        aggr_metrics_train, aggr_imgs_train = trainer.train_epoch(epoch)
        epoch_end_time = time.time()

        log_training(
            epoch=epoch,
            aggr_metrics_train=aggr_metrics_train,
            aggr_imgs_train=aggr_imgs_train,
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
                aggr_metrics_val, aggr_imgs_val = val_evaluator.evaluate(epoch)

            for k, v in aggr_metrics_val.items():
                tb_writer.add_scalar(f"{k}/val", v, epoch)

            # plot covariance matrix
            for k, v in aggr_imgs_val.items():
                tb_writer.add_figure(f"{k}/val", v, epoch)

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
                path=config["checkpoint_dir"],
                better=better,
            )

        if epoch % config["plot_interval"] == 0:
            plot_2d(
                method="pca",
                encoder=model.encoder,
                data_loader=train_loader,
                device=device,
                config=config,
                fname="pca_train.png",
                tb_writer=tb_writer,
                mode="train",
                epoch=epoch,
                num_classes=config.num_classes
                if "num_classes" in config.keys() and config.multilabel is False
                else 1,
                model="ts2vec",
                patch_len=config.patch_len,
                stride=config.stride,
            )
            plot_classwise_distribution(
                encoder=model.encoder,
                data_loader=train_loader,
                device=device,
                d_model=config.enc_d_model,
                num_classes=config.num_classes
                if "num_classes" in config.keys() and config.multilabel is False
                else 1,
                tb_writer=tb_writer,
                mode="train",
                epoch=epoch,
                model="ts2vec",
                patch_len=config.patch_len,
                stride=config.stride,
            )
            plot_2d(
                method="pca",
                encoder=model.encoder,
                data_loader=val_loader,
                device=device,
                config=config,
                fname="pca_val.png",
                tb_writer=tb_writer,
                mode="val",
                epoch=epoch,
                num_classes=config.num_classes
                if "num_classes" in config.keys() and config.multilabel is False
                else 1,
                model="ts2vec",
                patch_len=config.patch_len,
                stride=config.stride,
            )
            plot_classwise_distribution(
                encoder=model.encoder,
                data_loader=val_loader,
                device=device,
                d_model=config.enc_d_model,
                num_classes=config.num_classes
                if "num_classes" in config.keys() and config.multilabel is False
                else 1,
                tb_writer=tb_writer,
                mode="val",
                epoch=epoch,
                model="ts2vec",
                patch_len=config.patch_len,
                stride=config.stride,
            )

        else:
            save_checkpoint(
                epoch=epoch,
                model=trainer.model,
                optimizer=trainer.optimizer,
                path=config["checkpoint_dir"],
                better=False,
            )

        if patience_count > config.patience:
            break

    # load best model, iterate over test loader, get representations
    # mean over time axis, apply pca, then plot

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