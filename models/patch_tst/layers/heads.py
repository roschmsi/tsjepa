# Reference: https://github.com/yuqinie98/PatchTST

import torch
from torch import nn


class ClassificationTokenHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout, n_patch=None):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        # extract class token
        x = x[:, :, :, 0]
        x = self.flatten(x)
        # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)
        # y: bs x n_classes
        return y


class ClassificationFlattenHead(nn.Module):
    def __init__(self, n_vars, d_model, n_patch, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model * n_patch, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = self.flatten(x)  # x: bs x nvars * d_model * num_patch
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y


class ClassificationPoolHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x.mean(-1)
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(
        self,
        individual,
        n_vars,
        d_model,
        num_patch,
        forecast_len,
        head_dropout=0,
        flatten=False,
    ):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * num_patch]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)  # x: [bs x nvars x forecast_len]
        return x.transpose(2, 1)  # [bs x forecast_len x nvars]


class PatchRevinHead(nn.Module):
    def __init__(
        self,
        n_vars,
        d_model,
        input_num_patch,
        output_num_patch,
        patch_len,
        forecast_len,
        head_dropout=0,
        flatten=False,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * input_num_patch

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_val = nn.Linear(head_dim, forecast_len)
        self.linear_mean = nn.Linear(input_num_patch, output_num_patch)
        self.linear_std = nn.Linear(input_num_patch, output_num_patch)
        self.dropout = nn.Dropout(head_dropout)
        self.output_num_patch = output_num_patch
        self.patch_len = patch_len

    def forward(self, x, mean, std):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
        x = self.dropout(x)
        y = self.linear_val(x)  # x: [bs x nvars x forecast_len]

        y_mean = self.linear_mean(mean.squeeze(-1).transpose(1, 2))
        y_mean = y_mean.repeat_interleave(dim=-1, repeats=48)
        y_std = self.linear_std(std.squeeze(-1).transpose(1, 2))
        y_std = y_std.repeat_interleave(dim=-1, repeats=48)

        y = (y * y_std) + y_mean

        return y.transpose(2, 1)  # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
        return x
