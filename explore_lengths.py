import torch

from data.ecg_dataset import ECGDataset, load_and_split_dataframe
from options import Options
from utils import (
    seed_everything,
    setup,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    args = Options().parse()
    print("parsed")
    config = setup(args)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build ecg data
    train_df, val_df, test_df = load_and_split_dataframe(
        subset=config.data.subset, debug=config.debug
    )

    train_dataset = ECGDataset(
        train_df,
        window=config.data.window,
        num_windows=config.data.num_windows_train,
        src_path=config.data.dir,
        filter_bandwidth=config.data.filter_bandwidth,
        fs=config.data.fs,
    )
    lengths = []
    for data in tqdm(iter(train_dataset)):
        lengths.append(data[0].shape[0])

    plt.hist(lengths, bins=50)
    plt.savefig("train_dataset_lengths.jpg")
    plt.close()

    lengths = np.array(lengths)
    min = lengths.min()
    max = lengths.max()
    median = np.median(lengths)
    mean = lengths.mean()

    print("min:", min)
    print("max:", max)
    print("median:", median)
    print("mean:", mean)

    print("smaller than 2500:", (lengths < 2500).sum())
    print("larger than 2500:", (lengths > 2500).sum())

    plt.hist(lengths, bins=50, range=(0, 30000))
    plt.savefig("train_dataset_lengths_0_30000.jpg")
    plt.close()

    plt.hist(lengths, bins=50, range=(0, 50000))
    plt.savefig("train_dataset_lengths_0_50000.jpg")
    plt.close()

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
