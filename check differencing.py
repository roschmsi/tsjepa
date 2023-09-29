# Reference: https://github.com/gzerveas/mvts_transformer
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
from models.ts_jepa.utils import plot_2d, plot_classwise_distribution, plot_forecast
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

import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")


if __name__ == "__main__":
    X = torch.tensor([1, 5, 2, 19, 5, 13, 4]).reshape((1, -1, 1))
    targets = torch.tensor([7, 8, 2]).reshape((1, -1, 1))

    lag = 5

    X_orig = X
    targets_orig = targets

    pred_start = []

    targets = torch.cat([X_orig[:, -lag:, :], targets], dim=1)
    print("targets", targets)

    for i in range(lag):
        X = torch.diff(X, dim=1)
        pred_start.append(targets[:, :1, :])
        print("pred_start:", pred_start)
        targets = torch.diff(targets, dim=1)
        print("targets", targets)

    pred_start = pred_start[::-1]

    predictions = targets
    for i in range(lag):
        predictions = torch.cat([pred_start[i], predictions], dim=1)
        print("predictions", predictions)
        predictions = torch.cumsum(predictions, dim=1)
        print("predictions", predictions)

    predictions = predictions[:, -targets.shape[1] :, :]

    print((predictions - targets_orig).abs().sum())
