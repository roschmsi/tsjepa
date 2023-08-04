import glob
import os

import numpy as np
import torch

# from transform import *
from torch.utils.data import Dataset


def load_eeg_dataset(config):
    train_dataset = EEGDataset(
        root_dir=config.data_dir,
        eeg_channel=config.eeg_channel,
        target_idx=config.target_idx,
        set="train",
        debug=config.debug,
    )
    val_dataset = EEGDataset(
        root_dir=config.data_dir,
        eeg_channel=config.eeg_channel,
        target_idx=config.target_idx,
        set="val",
        debug=config.debug,
    )
    test_dataset = EEGDataset(
        root_dir=config.data_dir,
        eeg_channel=config.eeg_channel,
        target_idx=config.target_idx,
        set="test",
        debug=config.debug,
    )

    if config.debug:
        val_dataset = train_dataset
        test_dataset = train_dataset

    return train_dataset, val_dataset, test_dataset


class EEGDataset(Dataset):
    def __init__(
        self,
        root_dir,  # sleep_edf/
        eeg_channel,
        target_idx,
        set,
        debug=False,
    ):
        self.set = set
        self.sr = 100
        self.fold = 1
        self.num_splits = 10
        self.dset_name = "Sleep-EDF-2018"
        self.root_dir = root_dir
        self.debug = debug

        self.eeg_channel = eeg_channel

        self.seq_len = 1
        self.target_idx = target_idx

        # self.training_mode = "pretrain"

        self.dataset_path = os.path.join(self.root_dir, "npz")
        self.inputs, self.labels, self.epochs = self.split_dataset()

        if self.debug:
            self.epochs = self.epochs[:100]

        # if self.training_mode == "pretrain":
        #     self.transform = Compose(
        #         transforms=[
        #             RandomAmplitudeScale(),
        #             RandomTimeShift(),
        #             RandomDCShift(),
        #             RandomZeroMasking(),
        #             RandomAdditiveGaussianNoise(),
        #             RandomBandStopFilter(),
        #         ]
        #     )
        #     self.two_transform = TwoTransform(self.transform)

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx : idx + seq_len]

        # if self.set == "train":
        #     if self.training_mode == "pretrain":
        #         assert seq_len == 1
        #         input_a, input_b = self.two_transform(inputs)
        #         input_a = torch.from_numpy(input_a).float()
        #         input_b = torch.from_numpy(input_b).float()
        #         inputs = [input_a, input_b]
        #     elif self.training_mode in ["scratch", "fullyfinetune", "freezefinetune"]:
        #         inputs = inputs.reshape(1, n_sample)
        #         inputs = torch.from_numpy(inputs).float()
        #     else:
        #         raise NotImplementedError
        # else:
        #     if not self.training_mode == "pretrain":
        #         inputs = inputs.reshape(1, n_sample)
        #     inputs = torch.from_numpy(inputs).float()

        # inputs = inputs.transpose(0, 1)
        inputs = torch.from_numpy(inputs).transpose(0, 1).float()

        labels = self.labels[file_idx][idx : idx + seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]

        return inputs, labels

    def split_dataset(self):
        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [
            os.path.basename(x)
            for x in sorted(glob.glob(os.path.join(data_root, "*.npz")))
        ]
        data_fname_dict = {"train": [], "test": [], "val": []}
        split_idx_list = np.load(
            os.path.join(self.dataset_path, "idx_Sleep-EDF-2018.npy"),
            allow_pickle=True,
        )

        assert len(split_idx_list) == self.num_splits

        if self.dset_name == "Sleep-EDF-2013":
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict["test"].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict["val"].append(data_fname_list[i])
                else:
                    data_fname_dict["train"].append(data_fname_list[i])

        elif self.dset_name == "Sleep-EDF-2018":
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])

        elif (
            self.dset_name == "MASS"
            or self.dset_name == "Physio2018"
            or self.dset_name == "SHHS"
        ):
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))

        for data_fname in data_fname_dict[self.set]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file["x"])
            labels.append(npz_file["y"])
            seq_len = self.seq_len
            if self.dset_name == "MASS" and (
                "-02-" in data_fname or "-04-" in data_fname or "-05-" in data_fname
            ):
                seq_len = int(self.seq_len * 1.5)
            for i in range(len(npz_file["y"]) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1

        return inputs, labels, epochs
