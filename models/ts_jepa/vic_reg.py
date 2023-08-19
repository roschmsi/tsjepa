import torch
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_fn(z_enc, z_pred):
    num_features = z_enc.shape[-1]

    z_enc = z_enc.reshape(-1, num_features)
    z_pred = z_pred.reshape(-1, num_features)

    z_enc = z_enc - z_enc.mean(dim=0)
    z_pred = z_pred - z_pred.mean(dim=0)

    # TODO fine for 50 % masking, but actually problematic if different
    # should be better to concatenate the two and then compute the std
    std_enc = torch.sqrt(z_enc.var(dim=0) + 0.0001)
    std_pred = torch.sqrt(z_pred.var(dim=0) + 0.0001)

    std_loss = (z_enc.shape[0] / (z_enc.shape[0] + z_pred.shape[0])) * torch.mean(
        F.relu(1 - std_enc)
    ) + (z_pred.shape[0] / (z_enc.shape[0] + z_pred.shape[0])) * torch.mean(
        F.relu(1 - std_pred)
    )

    cov_enc = (z_enc.T @ z_enc) / (z_enc.shape[0] - 1)
    cov_pred = (z_pred.T @ z_pred) / (z_pred.shape[0] - 1)
    cov_loss = (z_enc.shape[0] / (z_enc.shape[0] + z_pred.shape[0])) * off_diagonal(
        cov_enc
    ).pow_(2).sum().div(num_features) + (
        z_pred.shape[0] / (z_enc.shape[0] + z_pred.shape[0])
    ) * off_diagonal(
        cov_pred
    ).pow_(
        2
    ).sum().div(
        num_features
    )

    return std_loss, cov_loss, cov_enc, cov_pred


def enc_vicreg_fn(z_enc):
    num_features = z_enc.shape[-1]

    z_enc = z_enc.reshape(-1, num_features)
    z_enc = z_enc - z_enc.mean(dim=0)

    std_enc = torch.sqrt(z_enc.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_enc))

    cov_enc = (z_enc.T @ z_enc) / (z_enc.shape[0] - 1)
    cov_loss = off_diagonal(cov_enc).pow_(2).sum().div(num_features)

    return std_loss, cov_loss
