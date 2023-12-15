import torch
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vibcreg(z):
    num_features = z.shape[-1]

    z = z.reshape(-1, num_features)
    z = z - z.mean(dim=0)

    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_mean = std.mean()
    std_loss = torch.mean(F.relu(1 - std))

    z_norm = F.normalize(z, p=2, dim=0)
    cov = z_norm.T @ z_norm
    cov_loss = off_diagonal(cov).pow_(2).mean()

    return std_loss, cov_loss, std_mean, cov


def vicreg(z):
    num_features = z.shape[-1]

    z = z.reshape(-1, num_features)
    z = z - z.mean(dim=0)

    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_mean = std.mean()
    std_loss = torch.mean(F.relu(1 - std))

    cov = (z.T @ z) / (z.shape[0] - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum().div(num_features)

    return std_loss, cov_loss, std_mean, cov
