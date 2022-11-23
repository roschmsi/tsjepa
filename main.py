"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

import os
import sys
import time

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Project modules
from options import Options
from loss import get_loss
from running import validate
from utils import (
    setup,
    seed_everything,
    load_model,
    save_model,
    count_parameters,
    log_training,
    readable_time,
)
from factory import model_factory, optimizer_factory, pipeline_factory
from physionet_evaluation.evaluate_12ECG_score import (
    load_weights,
    compute_challenge_metric,
)

# from loss import get_loss
from data.dataset import ECGDataset, load_and_split_dataframe, classes, normal_class

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

    # build ecg data
    train_df, val_df, test_df = load_and_split_dataframe(debug=config.debug)
    if config.debug:
        config.training.batch_size = 1
        config.val_interval = 1000

    train_dataset = ECGDataset(
        train_df,
        window=config.data.window,
        num_windows=config.data.num_windows_train,
        src_path=config.data.dir,
        filter_bandwidth=config.data.filter_bandwidth,
        fs=config.data.fs,
    )
    val_dataset = ECGDataset(
        val_df,
        window=config.data.window,
        num_windows=config.data.num_windows_val,
        src_path=config.data.dir,
        filter_bandwidth=config.data.filter_bandwidth,
        fs=config.data.fs,
    )
    test_dataset = ECGDataset(
        test_df,
        window=config.data.window,
        num_windows=config.data.num_windows_test,
        src_path=config.data.dir,
        filter_bandwidth=config.data.filter_bandwidth,
        fs=config.data.fs,
    )

    # create model
    logger.info("Creating model ...")
    model = model_factory(config)

    # freeze all weights except for output layer in classification task
    if config.model.name == "unsupervised_transformer":
        if config.model.freeze:
            for name, param in model.named_parameters():
                if name.startswith("output_layer"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(count_parameters(model, trainable=True))
    )

    optimizer = optimizer_factory(config, model)

    # options to continue training from previous model
    start_epoch = 0
    lr_step = 0

    # load model and optimizer states
    if config.load_model:
        model, optimizer, start_epoch = load_model(
            model,
            config["load_model"],  # load weights
            optimizer,
            config["resume"],  # load starting epoch and optimizer
            config["change_output"],  # finetuning on different task
            config.training["lr"],
            config.training["lr_step"],
            config.training["lr_factor"],
        )
    model.to(device)

    # initialize loss
    loss_module = get_loss(config)

    # initialize data generator and runner
    dataset_class, collate_fn, runner_class = pipeline_factory(config)

    max_len = config.data.window * config.data.fs

    if config.test:  # Only evaluate and skip training
        test_dataset = dataset_class(test_dataset)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=max_len),
        )
        test_evaluator = runner_class(
            model,
            test_loader,
            device,
            loss_module,
            print_interval=config["print_interval"],
            console=config["console"],
        )

        aggr_metrics_test, _ = test_evaluator.evaluate(keep_all=True)

        print_str = "Test Summary: "
        for k, v in aggr_metrics_test.items():
            print_str += "{}: {:8f} | ".format(k, v)
        logger.info(print_str)

        return

    # start model training
    train_dataset = dataset_class(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=max_len),
    )
    trainer = runner_class(
        model,
        train_loader,
        device,
        loss_module,
        optimizer,
        l2_reg=None,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    val_dataset = dataset_class(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=max_len),
    )
    val_evaluator = runner_class(
        model,
        val_loader,
        device,
        loss_module,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    tb_writer = SummaryWriter(config.output_dir)

    # initialize with +inf or -inf depending on key metric
    best_value = 1e16
    patience_count = 0

    # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    metrics = []
    best_metrics = {}

    # Evaluate on validation before training
    # aggr_metrics_val, best_metrics, best_value = validate(
    #     val_evaluator, tb_writer, config, best_metrics, best_value, epoch=0
    # )
    # metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    # metrics.append(list(metrics_values))

    logger.info("Starting training...")

    for epoch in tqdm(
        range(start_epoch + 1, config.training.epochs + 1),
        desc="Training Epoch",
        leave=False,
    ):
        mark = epoch if config["save_all"] else "last"

        # train the runner
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

        # evaluate if first or last epoch or at specified interval
        if (
            (epoch == config.training.epochs)
            or (epoch == start_epoch + 1)
            or (epoch % config["val_interval"] == 0)
        ):
            prev_best_value = best_value
            aggr_metrics_val, best_metrics, best_value = validate(
                val_evaluator,
                tb_writer,
                config,
                best_metrics,
                best_value,
                epoch,
            )

            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

            if best_value < prev_best_value:
                patience_count = 0
            else:
                patience_count += config["val_interval"]

        save_model(
            os.path.join(config.checkpoint_dir, "model_{}.pth".format(mark)),
            epoch,
            model,
            optimizer,
        )

        if patience_count > config.training.patience:
            break

        # specify for which models scheduling is required
        # learning rate scheduling
        scheduling = False
        if scheduling:
            if epoch == config["lr_step"][lr_step]:
                save_model(
                    os.path.join(config.checkpoint_dir, "model_{}.pth".format(epoch)),
                    epoch,
                    model,
                    optimizer,
                )
                lr = config.training.lr * config["lr_factor"][lr_step]
                if (
                    lr_step < len(config["lr_step"]) - 1
                ):  # so that this index does not get out of bounds
                    lr_step += 1
                logger.info("Learning rate updated to: ", lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

    # load best model, compute physionet challenge metric
    step = 0.02
    scores = []
    weights = load_weights(config.evaluation.weights_file, classes)
    model = load_model(
        model, model_path=os.path.join(config["checkpoint_dir"], "model_best.pth")
    )

    for thr in np.arange(0.0, 1.0, step):
        lbls = []
        probs = []

        for batch in val_loader:

            X, targets, padding_masks = batch
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)

            predictions = model(X, padding_masks)
            prob = predictions.sigmoid().data.cpu().numpy()
            probs.append(prob)
            lbls.append(targets.data.cpu().numpy())

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)

        preds = (probs > thr).astype(np.int)
        challenge_metric = compute_challenge_metric(
            weights, lbls, preds, classes, normal_class
        )
        scores.append(challenge_metric)

    # Best thrs and preds
    scores = np.array(scores)
    idxs = np.argmax(scores, axis=0)
    thrs = np.array([idxs * step])
    preds = (probs > thrs).astype(np.int)

    logger.info("Best loss: {}. Other metrics: {}".format(best_value, best_metrics))
    logger.info("Best challenge score: {}. Threshold: {}".format(scores[idxs], thrs[0]))
    logger.info("All Done!")

    total_runtime = time.time() - total_start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(total_runtime)
        )
    )


if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    main(config)
