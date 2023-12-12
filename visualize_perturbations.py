# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time
from functools import partial

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

from tsaug import Pool, Dropout

import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
import matplotlib

matplotlib.rcParams.update({"font.size": 12})
plt.rcParams["figure.dpi"] = 300


def main(config):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_epoch_time = 0
    total_start_time = time.time()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))

    # build data
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    if config.seq_len is not None:
        max_seq_len = config.seq_len
    else:
        max_seq_len = config.window * config.fs

    if config.use_patch:
        num_patch = (max_seq_len - config.patch_len) // config.stride + 1

    # initialize data generator and runner
    # channel independence, TODO solve for forecasting all dataset
    dataset_class = JEPADataset

    # prepare dataloader
    # train_dataset = dataset_class(train_dataset)
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False if config.debug else True,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     # collate_fn=mask_collator,
    # )
    # val_dataset = dataset_class(val_dataset)
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     # collate_fn=mask_collator,
    # )
    test_dataset = dataset_class(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        # collate_fn=mask_collator,
    )

    output_path = "/home/stud/roschman/ECGAnalysis/output/visualization_perturbations"

    for ts_id in [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
        X_full, y = test_dataset[ts_id]

        for i in range(X_full.shape[-1]):
            X = X_full[:, i]

            plt.figure(figsize=(10, 5))
            X_pert = X + torch.randn_like(X) * 0.1
            plt.plot(X_pert, c="blue", label="std=0.1")
            X_pert = X + torch.randn_like(X) * 0.3
            plt.plot(X_pert, c="green", label="std=0.3")
            plt.plot(X, c="red", label="original")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_path, f"perturbations_gaussian_noise_{ts_id}_{i}.png"
                )
            )
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(X, c="red", label="original")
            pool = Pool(size=10)
            X_pert = pool.augment(X.numpy())
            plt.plot(X_pert, c="blue", label="size=10")
            pool = Pool(size=50)
            X_pert = pool.augment(X.numpy())
            plt.plot(X_pert, c="green", label="size=50")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_path, f"perturbations_pool_{ts_id}_{i}.png")
            )
            plt.close()

            # plt.figure(figsize=(50, 10))
            # mean = X.mean().item()
            # plt.plot(X, c="red", label="original")
            # dropout = Dropout(p=0.01, fill=mean)
            # X_pert = dropout.augment(X.numpy())
            # plt.plot(X_pert, c="blue", label="size=2")
            # dropout = Dropout(p=0.05, fill=mean)
            # X_pert = dropout.augment(X.numpy())
            # plt.plot(X_pert, c="green", label="size=10")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig("perturbations_dropout.png")
            # plt.close()


if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    main(config)
