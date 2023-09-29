from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.patch_tst.layers.pos_encoding import positional_encoding
import torch


class SyntheticData(Dataset):
    def __init__(
        self, seq_len, pred_len, function, split, d_model, debug=False, scale=True
    ):
        super().__init__()
        self.scale = scale
        self.scaler = StandardScaler()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.debug = debug

        samples = 500000

        assert split in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[split]

        border1s = [
            0,
            int(0.6 * samples) - self.seq_len,
            int(0.8 * samples) - self.seq_len,
        ]
        border2s = [
            int(0.6 * samples),
            int(0.8 * samples),
            samples,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.timestamp = np.arange(samples).reshape((samples, 1))

        if function == "linear":
            self.data = np.apply_along_axis(
                func1d=lambda x: 0.5 * x, axis=0, arr=self.timestamp
            )
        elif function == "quadratic":
            self.data = np.apply_along_axis(
                func1d=lambda x: x**2, axis=0, arr=self.timestamp
            )
        elif function == "sin":
            self.data = np.apply_along_axis(
                func1d=lambda x: np.sin(x), axis=0, arr=self.timestamp
            )
        else:
            raise ValueError("Function not supported")

        if self.scale:
            train_data = self.data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data)

        # fixed absolute positional encoding as input
        self.pe = positional_encoding(
            pe="sincos", learn_pe=False, q_len=self.timestamp.shape[0], d_model=d_model
        )

        self.data = self.data[border1:border2]

    def __len__(self):
        if self.debug:
            return 2
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = torch.from_numpy(self.data[s_begin:s_end]).to(torch.float32)
        seq_y = torch.from_numpy(self.data[r_begin:r_end]).to(torch.float32)
        pe_x = self.pe[s_begin:s_end]
        pe_y = self.pe[r_begin:r_end]

        return seq_x, seq_y, pe_x, pe_y


def load_synthetic_dataset(config):
    train_dataset = SyntheticData(
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        function="linear",
        split="train",
        d_model=config.d_model,
        debug=config.debug,
        scale=True,
    )

    val_dataset = SyntheticData(
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        function="linear",
        split="val",
        d_model=config.d_model,
        debug=config.debug,
        scale=True,
    )

    test_dataset = SyntheticData(
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        function="linear",
        split="test",
        d_model=config.d_model,
        debug=config.debug,
        scale=True,
    )

    return train_dataset, val_dataset, test_dataset
