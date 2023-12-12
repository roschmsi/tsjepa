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
    init_scheduler,
)
from models.ts_jepa.utils import (
    load_checkpoint,
    plot_2d,
    plot_classwise_distribution,
    save_checkpoint,
)
from options import Options
from runner.tsjepa import JEPARunner
from utils import log_training, readable_time, seed_everything, setup
from models.patch_tst.layers.revin import RevIN

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
        config.val_interval = 5
        config.augment = False
        config.dropout = 0

    # build data
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    if config.seq_len is not None:
        max_seq_len = config.seq_len
    else:
        max_seq_len = config.window * config.fs

    # create model
    encoder, predictor = init_model_pretraining(
        device=device,
        seq_len=max_seq_len,
        patch_size=config.patch_len,
        in_chans=1 if config.channel_independence else config.feat_dim,
        enc_embed_dim=config.enc_d_model,
        enc_depth=config.enc_num_layers,
        enc_num_heads=config.enc_num_heads,
        enc_mlp_ratio=config.enc_mlp_ratio,
        pred_embed_dim=config.dec_d_model,
        pred_depth=config.dec_num_layers,
        pred_num_heads=config.dec_num_heads,
        pred_mlp_ratio=config.dec_mlp_ratio,
        norm_layer=config.norm,
        drop_rate=config.dropout,
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate,
        output_norm=not config.no_output_norm,
        learn_pe=config.learn_pe,
    )

    if config.ema:
        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
    else:
        target_encoder = None

    # -- make data transforms
    if config.masking == "random":
        mask_collator = RandomMaskCollator(
            ratio=config.masking_ratio,
            input_size=max_seq_len,
            patch_size=config.patch_len,
            channel_independence=config.channel_independence,
        )
    elif config.masking == "block":
        mask_collator = BlockMaskCollator(
            ratio=config.masking_ratio,
            input_size=max_seq_len,
            patch_size=config.patch_len,
            channel_independence=config.channel_independence,
        )
    else:
        raise ValueError("Unknown masking type")

    # initialize data generator and runner
    if config.dataset == "forecasting_all":
        dataset_class = ConcatenatedDataset
    elif config.dataset != "forecasting_all" and config.channel_independence:
        dataset_class = partial(CIDataset, num_channels=config.feat_dim)
    else:
        dataset_class = JEPADataset

    # start model training
    train_dataset = dataset_class(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False if config.debug else True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=mask_collator,
    )
    val_dataset = dataset_class(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=mask_collator,
    )
    test_dataset = dataset_class(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=mask_collator,
    )

    # create optimizer and scheduler
    optimizer = init_optimizer_enc_pred(
        encoder,
        predictor,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
    )
    scheduler = init_scheduler(optimizer, config)

    ipe = len(train_loader)

    if config.ema:
        momentum_scheduler = (
            config.ema_start
            + i * (config.ema_end - config.ema_start) / (ipe * config.epochs)
            for i in range(int(ipe * config.epochs) + 1)
        )
    else:
        momentum_scheduler = None

    start_epoch = 0

    # load pretrained weights
    if config.resume or config.load_model:
        if config.resume:
            path = os.path.join(config["output_dir"], "checkpoints", "model_last.pth")
        elif config.load_model:
            path = os.path.join(config["output_dir"], "checkpoints", "model_best.pth")
        encoder, predictor, target_encoder, optimizer, epoch = load_checkpoint(
            path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
        )
        if config.resume:
            start_epoch = epoch
            if scheduler is not None:
                for _ in range(start_epoch):
                    scheduler.step()
            if momentum_scheduler is not None:
                for _ in range(start_epoch * ipe):
                    next(momentum_scheduler)

    if config.revin:
        revin = RevIN(
            num_features=1,  # channel independence
            affine=True if config.revin_affine else False,
            subtract_last=False,
        )
    else:
        revin = None

    runner_class = JEPARunner

    runner_class = partial(
        runner_class,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        revin=revin,
        device=device,
        ema=config.ema,
        vic_reg=config.vic_reg,
        pred_weight=config.pred_weight,
        std_weight=config.std_weight,
        cov_weight=config.cov_weight,
    )

    trainer = runner_class(
        dataloader=train_loader,
        optimizer=optimizer,
        momentum_scheduler=momentum_scheduler,
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
                encoder=trainer.encoder,
                predictor=trainer.predictor,
                target_encoder=trainer.target_encoder,
                optimizer=trainer.optimizer,
                path=config["checkpoint_dir"],
                better=better,
            )

        if epoch % config["plot_interval"] == 0:
            plot_2d(
                method="pca",
                encoder=encoder,
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
            )
            plot_classwise_distribution(
                encoder=encoder,
                data_loader=train_loader,
                device=device,
                d_model=config.enc_d_model,
                num_classes=config.num_classes
                if "num_classes" in config.keys() and config.multilabel is False
                else 1,
                tb_writer=tb_writer,
                mode="train",
                epoch=epoch,
            )
            plot_2d(
                method="pca",
                encoder=encoder,
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
            )
            plot_classwise_distribution(
                encoder=encoder,
                data_loader=val_loader,
                device=device,
                d_model=config.enc_d_model,
                num_classes=config.num_classes
                if "num_classes" in config.keys() and config.multilabel is False
                else 1,
                tb_writer=tb_writer,
                mode="val",
                epoch=epoch,
            )

        else:
            save_checkpoint(
                epoch=epoch,
                encoder=trainer.encoder,
                predictor=trainer.predictor,
                target_encoder=trainer.target_encoder,
                optimizer=trainer.optimizer,
                path=config["checkpoint_dir"],
                better=False,
            )

        if patience_count > config.patience:
            break

    # load best model, iterate over test loader, get representations
    # mean over time axis, apply pca, then plot

    # if not config.debug:
    #     path = os.path.join(config["output_dir"], "checkpoints", "model_best.pth")
    #     encoder, predictor, target_encoder, optimizer, _ = load_checkpoint(
    #         path,
    #         encoder=encoder,
    #         predictor=predictor,
    #         target_encoder=target_encoder,
    #         optimizer=optimizer,
    #     )

    #     plot_2d(
    #         method="pca",
    #         encoder=encoder,
    #         data_loader=test_loader,
    #         device=device,
    #         config=config,
    #         fname="pca_test.png",
    #         num_classes=config.num_classes
    #         if "num_classes" in config.keys() and config.multilabel is False
    #         else 1,
    #     )

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
