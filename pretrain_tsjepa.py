# Reference: https://github.com/gzerveas/mvts_transformer

import copy
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import JEPADataset, load_dataset
from models.ts_jepa.mask import RandomMaskCollator
from models.ts_jepa.setup import init_model, init_opt
from models.ts_jepa.utils import load_checkpoint, plot_2d, save_checkpoint
from options import Options
from runner.tsjepa import JEPARunner
from utils import log_training, readable_time, seed_everything, setup

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
    encoder, predictor = init_model(
        device=device,
        seq_len=max_seq_len,
        in_chans=config.feat_dim,  # 1
        patch_size=config.patch_len,
        enc_embed_dim=config.enc_d_model,
        enc_depth=config.enc_num_layers,
        enc_num_heads=config.enc_num_heads,
        enc_mlp_ratio=config.enc_mlp_ratio,
        pred_depth=config.dec_num_layers,
        pred_num_heads=config.dec_num_heads,
        pred_embed_dim=config.dec_d_model,
        pred_mlp_ratio=config.dec_mlp_ratio,
        drop_rate=config.dropout,
    )

    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- make data transforms
    mask_collator = RandomMaskCollator(
        ratio=config.masking_ratio,
        input_size=max_seq_len,
        patch_size=config.patch_len,
    )

    # initialize data generator and runner
    dataset_class = JEPADataset
    runner_class = JEPARunner

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
    optimizer = init_opt(encoder, predictor)
    scheduler = None

    ipe = len(train_loader)
    ipe_scale = 1.0

    momentum_scheduler = (
        config.ema_start
        + i * (config.ema_end - config.ema_start) / (ipe * config.epochs * ipe_scale)
        for i in range(int(ipe * config.epochs * ipe_scale) + 1)
    )

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

    trainer = runner_class(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        dataloader=train_loader,
        device=device,
        optimizer=optimizer,
        momentum_scheduler=momentum_scheduler,
        mixup=config.mixup,
        print_interval=config["print_interval"],
        console=config["console"],
        multilabel=config.multilabel,
        scheduler=scheduler,
    )

    val_evaluator = runner_class(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        dataloader=val_loader,
        device=device,
        mixup=config.mixup,
        print_interval=config["print_interval"],
        console=config["console"],
        multilabel=config.multilabel,
    )

    test_evaluator = runner_class(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        dataloader=test_loader,
        device=device,
        mixup=config.mixup,
        print_interval=config["print_interval"],
        console=config["console"],
        multilabel=config.multilabel,
    )

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
        # if "freeze" in config.keys():
        #     if config.freeze and epoch > config.freeze_epochs:
        #         for name, param in model.named_parameters():
        #             param.requires_grad = True

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
                encoder=trainer.encoder,
                predictor=trainer.predictor,
                target_encoder=trainer.target_encoder,
                optimizer=trainer.optimizer,
                path=config["checkpoint_dir"],
                better=better,
            )

            plot_2d(
                method="tsne",
                encoder=encoder,
                data_loader=train_loader,
                device=device,
                config=config,
                fname="tsne_train.png",
                tb_writer=tb_writer,
                mode="train",
                epoch=epoch,
            )

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
            )

            plot_2d(
                method="tsne",
                encoder=encoder,
                data_loader=val_loader,
                device=device,
                config=config,
                fname="tsne_val.png",
                tb_writer=tb_writer,
                mode="val",
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
            )

        if patience_count > config.patience:
            break

    # load best model, iterate over test loader, get representations
    # mean over time axis, then plot with tsne

    if not config.debug:
        path = os.path.join(config["output_dir"], "checkpoints", "model_best.pth")
        encoder, predictor, target_encoder, optimizer, _ = load_checkpoint(
            path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
        )

        plot_2d(
            method="tsne",
            encoder=encoder,
            data_loader=test_loader,
            device=device,
            config=config,
            fname="tsne_test.png",
        )

        plot_2d(
            method="pca",
            encoder=encoder,
            data_loader=test_loader,
            device=device,
            config=config,
            fname="pca_test.png",
        )

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
