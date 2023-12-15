from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.ecg_dataset import load_ecg_dataset
from options import Options
from utils_old import seed_everything, setup

plt.rcParams["figure.dpi"] = 300

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

    visualization_dir = Path("/usr/stud/roschman/ECGAnalysis/output/visualization")
    visualization_dir.mkdir(parents=True, exist_ok=True)

    for num, batch in enumerate(iter(train_dataset)):
        X, targets = batch

        X = np.expand_dims(X, 0)

        Path(visualization_dir / f"sample_{num}").mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(X.shape[2], 1, figsize=(20, 20))
        for i in range(2):
            signal = X[0, :, i]
            axs[i].plot(signal, c="red" if i == 0 else "blue")
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(visualization_dir / f"sample_{num}" / "original.png", dpi=500)
        plt.close()

    # two leads
    for num, batch in enumerate(iter(train_dataset)):
        X, targets = batch

        X = np.expand_dims(X, 0)

        Path(visualization_dir / f"sample_{num}").mkdir(parents=True, exist_ok=True)

        for i in range(2):
            fig, axs = plt.subplots(1, 1, figsize=(10, 1))
            signal = X[0, :, i]
            axs.plot(signal, c="red" if i == 0 else "blue")
            # fig.patch.set_visible(False)
            axs.axis("off")

            plt.tight_layout()
            plt.savefig(visualization_dir / f"sample_{num}" / f"lead_{i}.png", dpi=500)
            plt.close()
