# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os

import torch
from torch.utils.data import DataLoader


from data.ecg_dataset import classes, load_ecg_dataset, normal_class
from data.uea_dataset import load_uea_dataset
from data.fc_dataset import load_fc_dataset
from factory import model_factory, optimizer_factory, pipeline_factory, tune_factory
from loss import get_criterion
from options import Options
from physionet_evaluation.evaluate_12ECG_score import (
    compute_challenge_metric,
    load_weights,
)
from runner import validate_without_logging
from utils import (
    seed_everything,
    setup_tuning,
)

import torch.optim as optim
from ray import tune, air
from ray.air.checkpoint import Checkpoint
from ray.air import session
from ray.tune.stopper import TrialPlateauStopper

from easydict import EasyDict

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
# logger.info("Loading packages ...")


def train(config):
    config = EasyDict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)

    if config["debug"]:
        config.batch_size = 1
        config.val_interval = 1
        config.augment = False
        config.epochs = 100
    # build ecg data
    if config.dataset in ["ecg", "ptb-xl", "ptb-xl-1000", "ptb-xl-5000"]:
        train_dataset, val_dataset, test_dataset = load_ecg_dataset(config)
    elif config.dataset in ["insect_wingbeat", "phoneme_spectra"]:
        train_dataset, val_dataset, test_dataset, config_data = load_uea_dataset(
            config.data, debug=config.debug
        )
        config.data = config_data
    elif config.dataset in ["ettm1"]:
        train_dataset, val_dataset, test_dataset = load_fc_dataset(
            config.data, debug=config.debug
        )
    else:
        raise ValueError("Dataset type is not specified")

    # create model
    model = model_factory(config)

    # freeze all weights except for output layer in classification task
    if config.model_name == "transformer_finetuning":
        if config.freeze:
            for name, param in model.named_parameters():
                if name.startswith("head"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    optimizer = optimizer_factory(config, model)
    if config.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # options to continue training from previous model
    start_epoch = 0

    # load model and optimizer states
    # if config.load_model:
    #     model, optimizer, start_epoch = load_model(
    #         model,
    #         config["load_model"],  # load weights
    #         optimizer,
    #         config["resume"],  # load starting epoch and optimizer
    #         config["change_output"],  # finetuning on different task
    #         config.lr,
    #     )
    model.to(device)

    # initialize loss
    loss_module = get_criterion(config)

    # initialize data generator and runner
    dataset_class, collate_fn, runner_class = pipeline_factory(config)

    if "max_seq_len" in config.keys():
        max_len = config.max_seq_len
    else:
        max_len = config.window * config.fs

    if config.test:  # Only evaluate and skip training
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
            model,
            test_loader,
            device,
            loss_module,
            mixup=config.mixup,
            print_interval=config["print_interval"],
            console=config["console"],
            multilabel=config.multilabel,
        )

        aggr_metrics_test, _ = test_evaluator.evaluate()
        # print_str = "Test Summary: "
        # for k, v in aggr_metrics_test.items():
        #     print_str += "{}: {:8f} | ".format(k, v)
        # logger.info(print_str)

        return

    # start model training
    train_dataset = dataset_class(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=max_len),
    )
    trainer = runner_class(
        model,
        train_loader,
        device,
        loss_module,
        optimizer,
        mixup=config.mixup,
        print_interval=config["print_interval"],
        console=config["console"],
        multilabel=config.multilabel,
        scheduler=scheduler,
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
        model,
        val_loader,
        device,
        loss_module,
        mixup=config.mixup,
        print_interval=config["print_interval"],
        console=config["console"],
        multilabel=config.multilabel,
    )

    # tb_writer = SummaryWriter(config.output_dir)

    # initialize with +inf or -inf depending on key metric
    best_value = 1e16
    patience_count = 0

    # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    # aggr_metrics_val, best_metrics, best_value = validate(
    #     val_evaluator, tb_writer, config, best_metrics, best_value, epoch=0
    # )
    # metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    # metrics.append(list(metrics_values))

    # logger.info("Starting training...")

    for epoch in range(start_epoch + 1, config.epochs + 1):
        # train the runner
        metrics_train = trainer.train_epoch(epoch)

        # evaluate if first or last epoch or at specified interval
        # if epoch % config["val_interval"] == 0:
        prev_best_value = best_value
        aggr_metrics_val, best_metrics, best_value = validate_without_logging(
            val_evaluator,
            config,
            best_metrics,
            best_value,
            epoch,
        )

        metrics_val_names, metrics_val_values = zip(*aggr_metrics_val.items())

        if best_value < prev_best_value:
            patience_count = 0
        else:
            patience_count += 1

        os.makedirs("model", exist_ok=True)
        torch.save((model.state_dict(), optimizer.state_dict()), "model/checkpoint.pth")
        checkpoint = Checkpoint.from_directory("model")

        if len(metrics_val_values) == 3:
            session.report(
                dict(loss=metrics_val_values[1].item(), auroc=metrics_val_values[2]),
                checkpoint=checkpoint,
            )
        else:
            session.report(
                dict(loss=metrics_val_values[1].item()),
                checkpoint=checkpoint,
            )

        if patience_count > config.patience:
            break


def main(config, num_samples=25):
    config = tune_factory(config)
    # sched = ASHAScheduler()

    resources_per_trial = {
        "cpu": os.cpu_count() // torch.cuda.device_count(),
        "gpu": 1,
    }  # set this for GPUs
    stopper = TrialPlateauStopper(
        metric="loss", std=0.01, num_results=5, grace_period=5
    )
    tuner = tune.Tuner(
        tune.with_resources(train, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=None,
            num_samples=num_samples,
            # trial_name_creator=trial_name_creator,
        ),
        run_config=air.RunConfig(
            name=config["formatted_timestamp"],
            local_dir=f"/home/stud/roschman/ECGAnalysis/output/{config.model_name}",
            # stop=stopper,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="loss",
                num_to_keep=1,
            ),
        ),
        param_space=config,
    )

    tuner.fit()
    # results = tuner.fit()

    # with results.get_best_result().checkpoint.as_directory() as best_checkpoint_path:
    #     model = model_factory(EasyDict(results.get_best_result().config))
    #     best_checkpoint = torch.load(
    #         os.path.join(best_checkpoint_path, "checkpoint.pth")
    #     )
    #     model.load_state_dict(best_checkpoint[0])

    # # load best model, compute physionet challenge metric
    # step = 0.02
    # scores = []
    # weights = load_weights(config.weights_file, classes)

    # for thr in np.arange(0.0, 1.0, step):
    #     lbls = []
    #     probs = []

    #     for batch in val_loader:

    #         X, targets, padding_masks = batch
    #         X = X.to(device)
    #         targets = targets.to(device)
    #         padding_masks = padding_masks.to(device)

    #         predictions = model(X, padding_masks)
    #         prob = predictions.sigmoid().data.cpu().numpy()
    #         probs.append(prob)
    #         lbls.append(targets.data.cpu().numpy())

    #     lbls = np.concatenate(lbls)
    #     probs = np.concatenate(probs)

    #     preds = (probs > thr).astype(np.int)
    #     challenge_metric = compute_challenge_metric(
    #         weights, lbls, preds, classes, normal_class
    #     )
    #     scores.append(challenge_metric)

    # # Best thrs and preds
    # scores = np.array(scores)
    # idxs = np.argmax(scores, axis=0)
    # thrs = np.array([idxs * step])
    # preds = (probs > thrs).astype(np.int)

    # logger.info("Best loss: {}. Other metrics: {}".format(best_value, best_metrics))
    # logger.info("Best challenge score: {}. Threshold: {}".format(scores[idxs], thrs[0]))
    # logger.info("All Done!")


if __name__ == "__main__":
    seed_everything()
    options = Options()
    args = options.parse()
    config = setup_tuning(args)
    main(config)
