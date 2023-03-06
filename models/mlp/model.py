import torch.nn as nn


class TSMLP(nn.Module):
    def __init__(self, max_len, c_in, target_dim, dropout):
        super(TSMLP, self).__init__()
        self.linear1 = nn.Linear(max_len * c_in, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, target_dim)

        # self.pool = nn.MaxPool1d()
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout1d(dropout)
        self.flatten = nn.Flatten()

    def forward(self, x, padding_masks=None):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        x = self.dropout(self.relu(self.linear4(x)))
        x = self.linear5(x)

        return x
