from torch.utils.data import Dataset

from data.ecg_dataset import load_ecg_dataset
from data.fc_dataset import load_fc_dataset


def load_dataset(config):
    if config.dataset in ["ecg", "ptb-xl", "ecg5", "ecg_gender"]:
        return load_ecg_dataset(config)
    elif config.dataset in [
        "etth1",
        "etth2",
        "ettm1",
        "ettm2",
        "illness",
        "traffic",
        "weather",
        "electricity",
    ]:
        return load_fc_dataset(config)
    else:
        raise ValueError("Dataset type is not specified")


class SupervisedDataset(Dataset):
    """
    Wrapper for supervised forecasting or classification
    """

    def __init__(self, dataset):
        super(SupervisedDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, ind):
        X, y = self.dataset.__getitem__(ind)
        return X, y

    def __len__(self):
        return len(self.dataset)


class CIDataset(Dataset):
    """
    Dataset with channel independence for pre-training
    """

    def __init__(self, dataset, num_channels, debug=False):
        super(CIDataset, self).__init__()
        self.dataset = dataset
        self.num_channels = num_channels
        self.debug = debug

    def __getitem__(self, ind):
        series_ind = ind // self.num_channels
        channel_ind = ind % self.num_channels

        X, _ = self.dataset.__getitem__(series_ind)
        X = X[:, channel_ind].unsqueeze(-1)

        return X

    def __len__(self):
        if self.debug:
            return 2
        else:
            return len(self.dataset) * self.num_channels


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    xb = xb.unfold(
        dimension=1, size=patch_len, step=stride
    )  # xb: [bs x num_patch x n_vars x patch_len]
    return xb




