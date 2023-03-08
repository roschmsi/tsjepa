from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal
import pandas as pd
from data.augmentation import augment
import random


classes = sorted(
    [
        "270492004",
        "164889003",
        "164890007",
        "426627000",
        "713427006",
        "713426002",
        "445118002",
        "39732003",
        "164909002",
        "251146004",
        "698252002",
        "10370003",
        "284470004",
        "427172004",
        "164947007",
        "111975006",
        "164917005",
        "47665007",
        "59118001",
        "427393009",
        "426177001",
        "426783006",
        "427084000",
        "63593006",
        "164934002",
        "59931005",
        "17338001",
    ]
)

normal_class = "426783006"


def load_ecg_dataset(config):
    data_df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/data/records_stratified_10_folds_v2.csv",
        index_col=0,
    ).reset_index(drop=True)

    # filter for ptb-xl data
    if config.dataset == "ptb-xl":
        data_df = data_df[data_df["Patient"].str.contains("HR")].reset_index(drop=True)
    elif config.dataset == "ptb-xl-5000":
        data_df = data_df[data_df["Patient"].str.contains("HR")].reset_index(drop=True)
        data_df = data_df.sample(frac=5000 / len(data_df), random_state=42).reset_index(
            drop=True
        )
    elif config.dataset == "ptb-xl-1000":
        data_df = data_df[data_df["Patient"].str.contains("HR")].reset_index(drop=True)
        data_df = data_df.sample(frac=1000 / len(data_df), random_state=42).reset_index(
            drop=True
        )
    elif config.dataset == "ecg":
        pass
    else:
        raise ValueError("Subset not specified")

    train_df = data_df.sample(frac=0.8, random_state=42)
    data_df = data_df.drop(train_df.index)
    val_df = data_df.sample(frac=0.5, random_state=42)
    test_df = data_df.drop(val_df.index)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if config.debug:
        train_df = train_df[:1]
        val_df = train_df[:1]
        test_df = train_df[:1]

    config["filter_bandwidth"] = [3, 45]

    train_dataset = ECGDataset(
        train_df,
        window=config.window,
        src_path=config.data_dir,
        filter_bandwidth=config.filter_bandwidth,
        fs=config.fs,
        aug=config.augment,
    )
    val_dataset = ECGDataset(
        val_df,
        window=config.window,
        src_path=config.data_dir,
        filter_bandwidth=config.filter_bandwidth,
        fs=config.fs,
        aug=False,
    )
    test_dataset = ECGDataset(
        test_df,
        window=config.window,
        src_path=config.data_dir,
        filter_bandwidth=config.filter_bandwidth,
        fs=config.fs,
        aug=False,
    )

    return train_dataset, val_dataset, test_dataset


class ECGDataset(Dataset):
    def __init__(self, df, window, src_path, filter_bandwidth, fs, aug):
        """Return randome window length segments from ecg signal, pad if window is too large
        df: trn_df, val_df or tst_df
        window: ecg window length e.g 2500 (5 seconds)
        nb_windows: number of windows to sample from record
        """
        self.df = df
        self.window = window
        self.src_path = Path(src_path)
        self.filter_bandwidth = filter_bandwidth
        self.fs = fs
        self.augment = aug
        self.augmentation_prob = 0.5

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path / (row.Patient + ".hea"))
        data, hdr = load_challenge_data(filename, fs=self.fs)
        seq_len = data.shape[-1]  # get the length of the ecg sequence

        # Apply band pass filter
        if self.filter_bandwidth is not None:
            data = apply_filter(data, self.filter_bandwidth, fs=self.fs)

        data = normalize(data)
        lbl = row[classes].values.astype(np.int)

        # Add just enough padding to allow window
        # pad = np.abs(np.min([seq_len - self.window, 0]))
        # if pad > 0:
        #     data = np.pad(data, ((0,0),(0,pad+1)))
        #     seq_len = data.shape[-1] # get the new length of the ecg sequence

        max_start = seq_len - self.window * self.fs + 1
        max_start = max_start if max_start > 1 else 1
        start = np.random.randint(max_start, size=1)[
            0
        ]  # get start indices of ecg segment
        data = data[:, start : start + self.window * self.fs]

        if self.augment and random.random() < self.augmentation_prob:
            data = np.expand_dims(data.transpose(), 0)
            data = augment(
                data,
                length=data.shape[1],
                sample_rate=self.fs,
            )
            data = data.squeeze().transpose()

        return data.transpose(), lbl


def load_challenge_data(header_file, fs):
    with open(header_file, "r") as f:
        header = f.readlines()
    sampling_rate = int(header[0].split()[2])
    mat_file = header_file.replace(".hea", ".mat")
    x = loadmat(mat_file)
    recording = np.asarray(x["val"], dtype=np.float64)

    # Standardize sampling rate
    if sampling_rate > fs:
        recording = decimate(recording, int(sampling_rate / fs))
    elif sampling_rate < fs:
        recording = resample(
            recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1
        )

    return recording, header


def normalize(seq, smooth=1e-8):
    """Normalize each sequence between -1 and 1"""
    return (
        2
        * (seq - np.min(seq, axis=1)[None].T)
        / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T
        - 1
    )


def apply_filter(signal, filter_bandwidth, fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=filter_bandwidth,
        sampling_rate=fs,
    )
    return signal
