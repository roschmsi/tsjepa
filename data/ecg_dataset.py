"""Adapted from https://physionet.org/content/challenge-2020/1.0.2/"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biosppy.signals.tools import filter_signal
from scipy.io import loadmat
from scipy.signal import decimate, resample
from torch.utils.data import Dataset

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
    elif config.dataset == "ecg5":
        reduced_df = None
        # 5 classes, 100 samples each
        classes_5 = ["270492004", "164889003", "164890007", "426627000", "713427006"]
        # choose 100 samples from each class, make sure that you only select samples that correspond to exactly one class

        for cls in classes_5:
            not_cls = [c for c in classes_5 if c != cls]
            cls_df = data_df[(data_df[cls] == 1) & (data_df[not_cls].sum(axis=1) == 0)]
            if reduced_df is None:
                reduced_df = cls_df
            else:
                reduced_df = pd.concat([reduced_df, cls_df], ignore_index=True)
        for c in classes:
            if c not in classes_5:
                reduced_df = reduced_df.drop(columns=[c])
        data_df = reduced_df.reset_index(drop=True)
    elif config.dataset == "ecg" or config.dataset == "ecg_gender":
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
        train_df = train_df[:100]
        val_df = train_df[:100]
        test_df = train_df[:100]

    train_dataset = ECGDataset(
        train_df,
        window=config.window,
        src_path=config.data_dir,
        filter_bandwidth=config.filter_bandwidth,
        fs=config.fs,
        aug=config.augment,
        rand_ecg=config.rand_ecg,
        reduced_classes=True if config.dataset == "ecg5" else False,
        gender=True if config.dataset == "ecg_gender" else False,
    )
    val_dataset = ECGDataset(
        val_df,
        window=config.window,
        src_path=config.data_dir,
        filter_bandwidth=config.filter_bandwidth,
        fs=config.fs,
        aug=False,
        rand_ecg="",
        reduced_classes=True if config.dataset == "ecg5" else False,
        gender=True if config.dataset == "ecg_gender" else False,
    )
    test_dataset = ECGDataset(
        test_df,
        window=config.window,
        src_path=config.data_dir,
        filter_bandwidth=config.filter_bandwidth,
        fs=config.fs,
        aug=False,
        rand_ecg="",
        reduced_classes=True if config.dataset == "ecg5" else False,
        gender=True if config.dataset == "ecg_gender" else False,
    )

    return train_dataset, val_dataset, test_dataset


class ECGDataset(Dataset):
    def __init__(
        self,
        df,
        window,
        src_path,
        filter_bandwidth,
        fs,
        aug,
        rand_ecg,
        reduced_classes=False,
        gender=False,
    ):
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
        self.rand_ecg = rand_ecg
        self.reduced_classes = reduced_classes
        self.gender = gender

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path / (row.Patient + ".hea"))
        data, hdr = load_challenge_data(filename, fs=self.fs)
        seq_len = data.shape[-1]  # get the length of the ecg sequence

        male = row["Gender_Male"]

        # Apply band pass filter
        if self.filter_bandwidth:
            data = apply_filter(data, filter_bandwidth=[3, 45], fs=self.fs)

        data = normalize(data)

        cls = classes
        if self.reduced_classes:
            cls = ["270492004", "164889003", "164890007", "426627000", "713427006"]

        lbl = row[cls].values.astype(np.float)

        max_start = seq_len - self.window * self.fs + 1
        max_start = max_start if max_start > 1 else 1
        start = np.random.randint(max_start, size=1)[
            0
        ]  # get start indices of ecg segment
        data = data[:, start : start + self.window * self.fs]

        if data.shape[1] < self.window * self.fs:
            missing_len = self.window * self.fs - data.shape[1]
            data = np.concatenate([data, np.zeros((12, missing_len))], 1)

        if self.augment and random.random() < self.augmentation_prob:
            data = np.expand_dims(data.transpose(), 0)
            data = augment(
                data,
                length=data.shape[1],
                sample_rate=self.fs,
            )
            data = data.squeeze().transpose()
        elif self.rand_ecg != "" and random.random() < self.augmentation_prob:
            data = np.expand_dims(data.transpose(), 0)
            data = randaug(data, self.rand_ecg)
            data = data.squeeze().transpose()

        if self.gender:
            lbl = male

        data = data.transpose()

        data = torch.from_numpy(data)
        data = data.to(torch.float32)
        lbl = torch.from_numpy(lbl)
        lbl = lbl.to(torch.float32)

        return data, lbl


def load_challenge_data(header_file, fs):
    with open(header_file, "r") as f:
        header = f.readlines()
    sampling_rate = int(header[0].split()[2])
    mat_file = header_file.replace(".hea", ".mat")
    x = loadmat(mat_file)
    recording = np.asarray(x["val"], dtype=np.float32)

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
