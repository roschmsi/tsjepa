# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import CIDataset, load_dataset
from model.revin import BlockRevIN, RevIN
from model.encoder import TransformerEncoder
from model.predictor import get_predictor
from model.tsjepa import BERT, TS2VecEMA, TS2VecNoEMA
from utils import load_checkpoint
from model.setup import init_optimizer, init_scheduler
from utils import plot_classwise_distribution
from model.vic_reg import vicreg
from options import Options
from runner.tsjepa import TS2VecRunner
from utils import save_checkpoint
from utils_old import log_training, readable_time, seed_everything, setup

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

    # dataset with channel independence
    dataset_class = partial(CIDataset, num_channels=config.feat_dim, debug=config.debug)

    # prepare datset and dataloader
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

    # create encoder
    encoder = TransformerEncoder(
        num_patch=num_patch,
        patch_size=config.patch_len,
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

    # create predictor
    predictor = get_predictor(config, max_seq_len=max_seq_len)

    # iterations per epoch
    ipe = len(train_loader)

    # create ts-jepa model
    if config.bert or config.mae:
        model = BERT(
            encoder=encoder,
            predictor=predictor,
            predictor_type=config.predictor,
            embed_dim=config.enc_d_model,
        )
    elif config.no_ema:
        model = TS2VecNoEMA(
            encoder=encoder,
            predictor=predictor,
            predictor_type=config.predictor,
            device=device,
            average_top_k_layers=config.average_top_k_layers,
            normalize_targets=config.normalize_targets,
            targets_rep=config.targets_rep,
            targets_norm=config.targets_norm,
            embed_dim=config.enc_d_model,
        )
    else:
        model = TS2VecEMA(
            encoder=encoder,
            predictor=predictor,
            predictor_type=config.predictor,
            device=device,
            average_top_k_layers=config.average_top_k_layers,
            normalize_targets=config.normalize_targets,
            targets_norm=config.targets_norm,
            normalize_pred=config.normalize_pred,
            pred_norm=config.pred_norm,
            embed_dim=config.enc_d_model,
            ema_decay=config.ema_decay,
            ema_end_decay=config.ema_end_decay,
            ema_anneal_end_step=config.ema_anneal_end_step * ipe,
            skip_pos_embed=config.skip_pos_embed,
            skip_patch_embed=config.skip_patch_embed,
            targets_rep=config.targets_rep,
        )
    model.to(device)

    # create optimizer and scheduler
    optimizer = init_optimizer(
        model, lr=config.lr, weight_decay=config.weight_decay,
    )

    scheduler = init_scheduler(optimizer, config)

    start_epoch = 0

    # load pretrained weights
    if config.resume or config.load_model:
        if config.resume:
            if config.load_model is None:
                path = os.path.join(
                    config["output_dir"], "checkpoints", "model_last.pth"
                )
            else:
                path = os.path.join(config.load_model, "checkpoints", "model_last.pth")
        elif config.load_model:
            path = os.path.join(config.load_model, "checkpoints", "model_best.pth")

        model, optimizer, scheduler, epoch = load_checkpoint(
            path,
            model=model,
            optimizer=optimizer if config.resume else None,
            scheduler=scheduler if config.resume else None,
        )
        if config.resume:
            start_epoch = epoch

            for i in range(start_epoch * ipe):
                decay = model.ema.get_annealed_rate(
                    model.ema_decay,
                    model.ema_end_decay,
                    model.ema.num_updates,
                    model.ema_anneal_end_step,
                )
                model.ema.num_updates += 1
                model.ema.decay = decay

            # set decay rate
        #     if scheduler is not None:
        #         for _ in range(start_epoch):
        #             scheduler.step()

    # if config.load_encoder:
    #     path = os.path.join(config.load_encoder, "checkpoints", "model_best.pth")
    #     model.encoder, _, _, _ = load_checkpoint_encoder(path, model=model.encoder)
    #     model.ema.model, _, _, _ = load_checkpoint_encoder(path, model=model.ema.model)

    if config.revin:
        if config.masking == "random":
            revin = RevIN(
                num_features=1,  # channel independence
                affine=True if config.revin_affine else False,
                subtract_last=False,
            )
        elif config.masking == "block":
            revin = BlockRevIN(
                num_features=1,  # channel independence
                affine=True if config.revin_affine else False,
                subtract_last=False,
                masking_ratio=config.masking_ratio,
            )
        else:
            raise NotImplementedError
        # revin = revin.to(device)
    else:
        revin = None

    # criterion
    if config.loss == "l1":
        criterion = nn.L1Loss(reduction="mean")
    elif config.loss == "smoothl1":
        criterion = nn.SmoothL1Loss(reduction="mean", beta=config.smoothl1_beta)
    else:
        criterion = nn.MSELoss(reduction="mean")
    criterion.to(device)

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
        pred_weight=config.pred_weight,
        std_weight=config.std_weight,
        cov_weight=config.cov_weight,
        criterion=criterion,
        no_ema=config.no_ema,
        regfn=vicreg,
        embedding_space=not config.bert,
    )

    trainer = runner_class(
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    val_evaluator = runner_class(dataloader=val_loader)
    test_evaluator = runner_class(dataloader=test_loader)

    tb_writer = SummaryWriter(config.output_dir)

    if config.test:
        logger.info("Test performance:")
        aggr_metrics_test, aggr_imgs_test = test_evaluator.evaluate()
        for k, v in aggr_metrics_test.items():
            logger.info(f"{k}: {v}")
            tb_writer.add_scalar(f"{k}/test", v, epoch)

        for k, v in aggr_imgs_test.items():
            tb_writer.add_figure(f"{k}/test", v, epoch)

        plot_classwise_distribution(
            encoder=model.encoder,
            data_loader=test_loader,
            device=device,
            d_model=config.enc_d_model,
            num_classes=config.num_classes
            if "num_classes" in config.keys() and config.multilabel is False
            else 1,
            revin=revin,
            tb_writer=tb_writer,
            mode="test",
            epoch=epoch,
            model="ts2vec",
            patch_len=config.patch_len,
            stride=config.stride,
        )

        return

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
                scheduler=trainer.scheduler,
                path=config["checkpoint_dir"],
                better=better,
            )

        if epoch % config["plot_interval"] == 0:
            # plot_2d(
            #     method="pca",
            #     encoder=model.encoder,
            #     data_loader=train_loader,
            #     device=device,
            #     config=config,
            #     fname="pca_train.png",
            #     tb_writer=tb_writer,
            #     mode="train",
            #     epoch=epoch,
            #     num_classes=config.num_classes
            #     if "num_classes" in config.keys() and config.multilabel is False
            #     else 1,
            #     model="ts2vec",
            #     patch_len=config.patch_len,
            #     stride=config.stride,
            #     revin=revin,
            # )
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
                revin=revin,
            )
            # plot_2d(
            #     method="pca",
            #     encoder=model.encoder,
            #     data_loader=val_loader,
            #     device=device,
            #     config=config,
            #     fname="pca_val.png",
            #     tb_writer=tb_writer,
            #     mode="val",
            #     epoch=epoch,
            #     num_classes=config.num_classes
            #     if "num_classes" in config.keys() and config.multilabel is False
            #     else 1,
            #     model="ts2vec",
            #     patch_len=config.patch_len,
            #     stride=config.stride,
            #     revin=revin,
            # )
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
                revin=revin,
            )

        else:
            save_checkpoint(
                epoch=epoch,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
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
