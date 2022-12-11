from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tsaug import AddNoise, Crop, Drift, Dropout, Pool, Quantize, Reverse, TimeWarp

from data.ecg_dataset import load_ecg_dataset
from data.fc_dataset import load_fc_dataset
from data.uea_dataset import load_uea_dataset
from options import Options
from utils import seed_everything, setup

if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.debug:
        config.training.batch_size = 1
        config.val_interval = 1000

    config.data.augment = False

    # build ecg data
    if config.data.type == "ecg":
        train_dataset, val_dataset, test_dataset = load_ecg_dataset(config)
    elif config.data.type == "uea":
        train_dataset, val_dataset, test_dataset, config_data = load_uea_dataset(
            config.data, debug=config.debug
        )
        config.data = config_data
    elif config.data.type == "fc":
        train_dataset, val_dataset, test_dataset = load_fc_dataset(
            config.data, debug=config.debug
        )
    else:
        raise ValueError("Dataset type is not specified")

    visualization_dir = Path("/usr/stud/roschman/ECGAnalysis/output/visualization")
    visualization_dir.mkdir(parents=True, exist_ok=True)

    augmentations = {
        "noise": AddNoise(loc=0, scale=(0.1)),
        "crop": Crop(
            size=(
                int(0.25 * 1000),
                int(0.75 * 1000),
            ),
            resize=int(1000),
        ),
        "drift": Drift(max_drift=0.5, kind="multiplicative"),
        "dropout": Dropout(
            p=0.1,
            fill=0,
            size=[
                # int(0.001 * config.data.fs),
                int(0.01 * config.data.fs),
                int(0.1 * config.data.fs),
            ],
        ),
        "pool": Pool(size=[2, 3, 5]),
        "quantize": Quantize(n_levels=[10, 20, 30]),
        "reverse": Reverse(),
        "timewarp": TimeWarp(n_speed_change=3, max_speed_ratio=(2, 3)),
    }

    for num, batch in enumerate(iter(train_dataset)):
        X, targets = batch

        X = np.expand_dims(X, 0)

        Path(visualization_dir / f"sample_{num}").mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(X.shape[2], 1, figsize=(15, 50))
        for i in range(X.shape[2]):
            signal = X[0, :, i]
            axs[i].plot(signal)

        plt.savefig(visualization_dir / f"sample_{num}" / "original.png")
        plt.close()

        # (N, T, C), where T is the length of a series, N is
        #     the number of series, and C is the number of a channels in a series.

        for aug_name in augmentations.keys():
            augmentation = augmentations[aug_name]
            X_aug = augmentation.augment(X)
            fig, axs = plt.subplots(X_aug.shape[2], 1, figsize=(15, 50))
            for i in range(X_aug.shape[2]):
                signal = X_aug[0, :, i]
                axs[i].plot(signal)
            plt.savefig(visualization_dir / f"sample_{num}" / f"{aug_name}.png")
            plt.close()
