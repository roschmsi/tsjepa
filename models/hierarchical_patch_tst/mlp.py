import torch.nn as nn
import torch


class DownsamplingMLP(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, c_in, c_out, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * c_in, c_out)
        self.norm = norm_layer(win_size * c_in)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        x: bs, num_patch, d_model
        """
        batch_size, num_patch, d_model = x.shape
        pad_num = num_patch % self.win_size

        # TODO maybe just repeat the final values ?
        if pad_num != 0:
            # pad_num = self.win_size - pad_num
            # x = torch.cat((x[:, :pad_num, :], x), dim=1)
            # discard first few elements
            x = x[:, pad_num:, :]

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, i :: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear_trans(x)

        return x


class UpsamplingMLP(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, c_in, c_out, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.c_out = c_out
        self.win_size = win_size
        self.linear_trans = nn.Linear(c_in, win_size * c_out)
        self.norm = norm_layer(win_size * c_in)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        x: bs, num_patch, d_model
        """
        batch_size, num_patch, d_model = x.shape

        # x = self.norm(x)
        x = self.dropout(x)
        x = self.linear_trans(x)

        x = x.reshape(batch_size, num_patch, self.win_size, self.c_out)
        x = x.reshape(batch_size, num_patch * self.win_size, self.c_out)

        return x
