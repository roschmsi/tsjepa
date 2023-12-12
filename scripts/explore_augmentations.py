from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tsaug import AddNoise, Crop, Drift, Dropout, Pool, Quantize, Reverse

from data.ecg_dataset import load_ecg_dataset
from data.fc_dataset import load_fc_dataset
from options import Options
from utils import seed_everything, setup
from tsaug import AddNoise, Dropout, Pool

if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.debug:
        config.batch_size = 1
        config.val_interval = 1000

    config.augment = False

    # build ecg data
    if config.dataset == "ecg" or config.dataset == "ptb-xl":
        train_dataset, val_dataset, test_dataset = load_ecg_dataset(config)
    elif config.dataset == "uea":
        train_dataset, val_dataset, test_dataset, config_data = load_uea_dataset(
            config.data, debug=config.debug
        )
        config.data = config_data
    elif config.dataset == "fc":
        train_dataset, val_dataset, test_dataset = load_fc_dataset(
            config.data, debug=config.debug
        )
    else:
        raise ValueError("Dataset type is not specified")

    visualization_dir = Path("/usr/stud/roschman/ECGAnalysis/output/visualization")
    visualization_dir.mkdir(parents=True, exist_ok=True)

    augmentations = {
        "noise": AddNoise(loc=0, scale=(0.1, 0.2)),
        "crop": Crop(
            size=(
                int(0.8 * config.window * config.fs),
                int(1.0 * config.window * config.fs),
            ),
            resize=int(config.window * config.fs),
        ),
        "drift": Drift(max_drift=0.25, kind="multiplicative"),
        "dropout": Dropout(
            p=0.05,
            fill=0,
            size=[
                # int(0.001 * sample_rate),
                int(0.01 * config.fs),
                int(0.05 * config.fs),
                int(0.1 * config.fs),
            ],
        ),
        "pool": Pool(size=[2, 3, 5]),
        "quantize": Quantize(n_levels=[5, 10, 15]),
        "reverse": Reverse(),
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
