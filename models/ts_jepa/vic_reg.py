import torch
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vibcreg_fn(z_enc, z_pred):
    num_features = z_enc.shape[-1]

    z_enc = z_enc.reshape(-1, num_features)
    z_pred = z_pred.reshape(-1, num_features)

    z_enc = z_enc - z_enc.mean(dim=0)
    z_pred = z_pred - z_pred.mean(dim=0)

    frac_enc = z_enc.shape[0] / (z_enc.shape[0] + z_pred.shape[0])
    frac_pred = 1 - frac_enc

    # TODO fine for 50 % masking, but actually problematic if different
    # should be better to concatenate the two and then compute the std
    std_enc = torch.sqrt(z_enc.var(dim=0) + 0.0001)
    std_pred = torch.sqrt(z_pred.var(dim=0) + 0.0001)

    std_loss_enc = torch.mean(F.relu(1 - std_enc))
    std_loss_pred = torch.mean(F.relu(1 - std_pred))

    std_loss = frac_enc * std_loss_enc + frac_pred * std_loss_pred

    # cov_enc = (z_enc.T @ z_enc) / (z_enc.shape[0] - 1)
    # cov_pred = (z_pred.T @ z_pred) / (z_pred.shape[0] - 1)
    # cov_loss_enc = off_diagonal(cov_enc).pow_(2).sum().div(num_features)
    # cov_loss_pred = off_diagonal(cov_pred).pow_(2).sum().div(num_features)
    z_enc = F.normalize(z_enc, p=2, dim=0)
    z_pred = F.normalize(z_pred, p=2, dim=0)

    cov_enc = z_enc.T @ z_enc
    cov_pred = z_pred.T @ z_pred

    cov_enc.fill_diagonal_(0)
    cov_pred.fill_diagonal_(0)

    cov_loss_enc = (cov_enc**2).mean()
    cov_loss_pred = (cov_pred**2).mean()

    cov_loss = frac_enc * cov_loss_enc + frac_pred * cov_loss_pred

    return std_loss, cov_loss, cov_enc, cov_pred


def vicreg_fn(z_enc, z_pred):
    num_features = z_enc.shape[-1]

    z_enc = z_enc.reshape(-1, num_features)
    z_pred = z_pred.reshape(-1, num_features)

    z_enc = z_enc - z_enc.mean(dim=0)
    z_pred = z_pred - z_pred.mean(dim=0)

    frac_enc = z_enc.shape[0] / (z_enc.shape[0] + z_pred.shape[0])
    frac_pred = 1 - frac_enc

    std_enc = torch.sqrt(z_enc.var(dim=0) + 1e-6)
    std_loss_enc = torch.mean(F.relu(1 - std_enc))

    std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-6)
    std_loss_pred = torch.mean(F.relu(1 - std_pred))

    std_loss = frac_enc * std_loss_enc + frac_pred * std_loss_pred

    cov_enc = (z_enc.T @ z_enc) / (z_enc.shape[0] - 1)
    cov_pred = (z_pred.T @ z_pred) / (z_pred.shape[0] - 1)
    cov_loss_enc = off_diagonal(cov_enc).pow_(2).sum().div(num_features)
    cov_loss_pred = off_diagonal(cov_pred).pow_(2).sum().div(num_features)

    cov_loss = frac_enc * cov_loss_enc + frac_pred * cov_loss_pred

    return std_loss, cov_loss, cov_enc, cov_pred


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