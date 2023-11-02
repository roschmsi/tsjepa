# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.dataset import load_dataset

from data.ecg_dataset import classes, normal_class
from factory import (
    setup_model,
    setup_optimizer,
    setup_pipeline,
    setup_scheduler,
)
from loss import get_criterion

# from models.ts_jepa.utils import plot_2d, plot_classwise_distribution, plot_forecast
from options import Options
from evaluation.evaluate_12ECG_score import (
    compute_challenge_metric,
    load_weights,
)
from utils import (
    count_parameters,
    load_model,
    log_training,
    readable_time,
    save_model,
    seed_everything,
    setup,
)

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

    if config.debug:
        config.batch_size = 5
        config.val_interval = 5
        config.augment = False
        config.dropout = 0

    # build ecg data
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    # create model
    model = setup_model(config)

    if "freeze" in config.keys():
        if config.freeze:
            for name, param in model.named_parameters():
                if name.startswith("head"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(count_parameters(model, trainable=True))
    )

    # create optimizer and scheduler
    optimizer = setup_optimizer(config, model)

    # load model and optimizer states
    start_epoch = 0
    if config.resume or config.load_model:
        if config.resume:
            path = os.path.join(config["output_dir"], "checkpoints", "model_last.pth")
            model, optimizer, start_epoch = load_model(
                model,
                path,
                optimizer,
                resume=config["resume"],  # load starting epoch and optimizer
                change_output=config.finetuning,  # finetuning on different task
                device=device,
            )
        else:
            path = os.path.join(config["load_model"], "checkpoints", "model_best.pth")
            model = load_model(
                model,
                path,
                optimizer,
                resume=config["resume"],  # load starting epoch and optimizer
                change_output=config.finetuning,  # finetuning on different task
                device=device,
            )

    model.to(device)

    # initialize loss
    criterion = get_criterion(config)

    # initialize data generator and runner
    dataset_class, collate_fn, runner_class = setup_pipeline(config)

    if config.seq_len is not None:
        max_len = config.seq_len
    else:
        max_len = config.window * config.fs

    # start model training
    train_dataset = dataset_class(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False if config.debug else True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=max_len),
    )

    scheduler = setup_scheduler(config, optimizer, iters_per_epoch=len(train_loader))
    trainer = runner_class(
        model=model,
        dataloader=train_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    val_dataset = dataset_class(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=max_len),
    )
    val_evaluator = runner_class(
        model=model,
        dataloader=val_loader,
        device=device,
        criterion=criterion,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    test_dataset = dataset_class(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=max_len),
    )
    test_evaluator = runner_class(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    if config.test:
        logger.info("Test performance:")
        with torch.no_grad():
            aggr_metrics_test = test_evaluator.evaluate()
        for k, v in aggr_metrics_test.items():
            logger.info(f"{k}: {v}")

        return

    tb_writer = SummaryWriter(config.output_dir)

    patience_count = 0
    best_loss_val = 1e16
    best_metrics_val = {}

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
                best_metrics_val = aggr_metrics_val.copy()
                patience_count = 0
                save_model(
                    path=os.path.join(config["checkpoint_dir"], "model_best.pth"),
                    epoch=epoch,
                    model=val_evaluator.model,
                )
            else:
                patience_count += config["val_interval"]

            # plot_forecast(
            #     model=model,
            #     data_loader=train_loader,
            #     device=device,
            #     config=config,
            #     tb_writer=tb_writer,
            #     epoch=epoch,
            # )

            # plot_2d(
            #     method="pca",
            #     encoder=model.backbone,
            #     data_loader=train_loader,
            #     device=device,
            #     config=config,
            #     fname="pca_train.png",
            #     tb_writer=tb_writer,
            #     mode="train",
            #     epoch=epoch,
            #     num_classes=1,
            #     supervised=True,
            #     model=config.model_name,
            # )
            # plot_classwise_distribution(
            #     encoder=model.backbone,
            #     data_loader=train_loader,
            #     device=device,
            #     d_model=config.d_model,
            #     num_classes=1,
            #     tb_writer=tb_writer,
            #     mode="train",
            #     epoch=epoch,
            #     supervised=True,
            #     model=config.model_name,
            # )
            # plot_2d(
            #     method="pca",
            #     encoder=model.backbone,
            #     data_loader=val_loader,
            #     device=device,
            #     config=config,
            #     fname="pca_val.png",
            #     tb_writer=tb_writer,
            #     mode="val",
            #     epoch=epoch,
            #     num_classes=1,
            #     supervised=True,
            #     model=config.model_name,
            # )
            # plot_classwise_distribution(
            #     encoder=model.backbone,
            #     data_loader=val_loader,
            #     device=device,
            #     d_model=config.d_model,
            #     num_classes=1,
            #     tb_writer=tb_writer,
            #     mode="val",
            #     epoch=epoch,
            #     supervised=True,
            #     model=config.model_name,
            # )

        # save model every n epochs
        if epoch % 50 == 0:
            save_model(
                path=os.path.join(config.checkpoint_dir, f"model_{epoch}.pth"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
            )

        save_model(
            path=os.path.join(config.checkpoint_dir, "model_last.pth"),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
        )

        if patience_count > config.patience:
            break

    # final evaluation on test dataset
    model = load_model(
        model, path=os.path.join(config["checkpoint_dir"], "model_best.pth")
    )

    logger.info("Best validation performance:")
    for k, v in best_metrics_val.items():
        logger.info(f"{k}: {v}")

    with torch.no_grad():
        # test_evaluator.model = model
        aggr_metrics_test = test_evaluator.evaluate(
            epoch_num=int(best_metrics_val["epoch"])
        )

    logger.info("Best test performance:")
    for k, v in aggr_metrics_test.items():
        logger.info(f"{k}: {v}")

    # load best model, compute physionet challenge metric
    if config.dataset == "ecg" and config.task == "classification":
        logger.info("Compute PhysioNet 2020 challenge metric")
        step = 0.02
        scores = []
        weights = load_weights(config.weights_file, classes)

        for thr in np.arange(0.0, 1.0, step):
            lbls = []
            probs = []

            for batch in val_loader:
                X, targets, padding_masks = batch
                X = X.to(device)
                targets = targets.to(device)
                padding_masks = padding_masks.to(device)

                with torch.no_grad():
                    predictions = model(X, padding_masks)
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
            X, targets, padding_masks = batch
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)

            with torch.no_grad():
                predictions = model(X, padding_masks)
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
