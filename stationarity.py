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
    args = Options().parse()
    config = setup(args)
    seed_everything()
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    data_original = np.concatenate(
        [train_dataset.data_x, val_dataset.data_x, test_dataset.data_x], axis=0
    )

    lag = 1

    for d in range(config.feat_dim):
        data = data_original[:500, d]
        plt.figure(figsize=(200, 10))
        # plt.plot(data)

        data_orig = data

        pred_start = data[0]
        data = np.diff(data, n=2, axis=0)

        segments = []
        for i in range(24):
            segments.append(data[:, i::24, :])

        # # standard normalization
        # # data = (data - data.mean()) / data.std()

        # data_before_normalization = data

        # # scale between -1 and 1
        # min = -1
        # max = 1
        # max_min_diff = data.max() - data.min()
        # data_min = data.min()
        # data = (data - data_min) / max_min_diff
        # data = data * (max - min) + min

        # data = (data - min) / (max - min)
        # data = data * max_min_diff + data_min

        # assert abs(data - data_before_normalization).sum() < 1e-8

        # data = np.concatenate([np.array([pred_start]), data], axis=0)
        # data = np.cumsum(data, axis=0)

        # assert abs(data - data_orig).sum() < 1e-8

        # plt.plot(data)
        # data = np.diff(data, n=1, axis=0)
        # data = np.diff(data, n=1, axis=0)
        plt.plot(data)
        # data = np.log(data)
        # plt.plot(data)
        plt.tight_layout()
        plt.savefig(f"{config.dataset}_dim={d}_diff={2}.png")
