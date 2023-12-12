import logging
import os
import sys
import time
from functools import partial
import argparse

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
import matplotlib.pyplot as plt
import matplotlib
from easydict import EasyDict
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")


matplotlib.rcParams.update({"font.size": 12})
plt.rcParams["figure.dpi"] = 300


def main(config):
    # create config.output_dir directory if it does not exist

    path = config.output_dir
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    path_supervised = f"{config.supervised_model}/robustness.csv"
    path_finetuning = f"{config.tsjepa_finetuning}/robustness.csv"
    path_linear_probing = f"{config.tsjepa_linear_probing}/robustness.csv"

    performance_supervised = pd.read_csv(path_supervised)
    performance_finetuning = pd.read_csv(path_finetuning)
    performance_linear_probing = pd.read_csv(path_linear_probing)

    plt.figure(figsize=(8, 6))
    plt.xlabel("Gaussian noise std")
    plt.ylabel("MSE")

    plt.plot(
        performance_supervised["perturbation_std"],
        performance_supervised["mse"],
        label="supervised",
    )
    plt.plot(
        performance_finetuning["perturbation_std"],
        performance_finetuning["mse"],
        label="TS-JEPA finetuned",
    )
    plt.plot(
        performance_linear_probing["perturbation_std"],
        performance_linear_probing["mse"],
        label="TS-JEPA linear probing",
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/robustness_plot.png", dpi=300)
    plt.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline for all models")
    parser.add_argument("--supervised_model", type=str)
    parser.add_argument("--tsjepa_finetuning", type=str)
    parser.add_argument("--tsjepa_linear_probing", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    config = args.__dict__  # configuration dictionary
    config = EasyDict(config)
    main(config)
