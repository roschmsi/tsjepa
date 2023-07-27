import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def save_checkpoint(
    epoch,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    path,
    better,
):
    save_dict = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "opt": optimizer.state_dict(),
        # "lr": lr,
        # "loss": loss,
        # "batch_size": batch_size,
    }

    torch.save(save_dict, os.path.join(path, "model_last.pth"))
    # torch.save(save_dict, os.path.join(path, f"model_{epoch}.pth"))
    if better:
        torch.save(save_dict, os.path.join(path, "model_best.pth"))


def load_checkpoint(
    path,
    encoder,
    predictor,
    target_encoder,
    optimizer,
):
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["encoder"]
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint["target_encoder"]
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
            )

        # -- loading optimizer
        optimizer.load_state_dict(checkpoint["opt"])

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder, predictor, target_encoder, optimizer, epoch


def pca(X):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d


def tsne(X):
    tsne = TSNE(n_components=2, perplexity=1)  # TODO: tune perplexity
    X_2d = tsne.fit_transform(X)
    return X_2d


def plot_2d(
    method,
    encoder,
    data_loader,
    device,
    config,
    fname,
    tb_writer=None,
    mode=None,
    epoch=None,
):
    encoder = encoder.eval()
    X_rep = []
    y_rep = []

    with torch.no_grad():
        # change dataset class, mean across time axis and
        for X, y, masks_enc, masks_pred in data_loader:
            X = X.float().to(device)
            X_enc = encoder(X, masks=None)
            X_rep.append(X_enc.mean(dim=1))
            y_rep.append(torch.where(y == 1)[1])

        X_rep = torch.cat(X_rep, dim=0).cpu().detach().numpy()
        y_rep = torch.cat(y_rep, dim=0).numpy()

    if method == "pca":
        X_2d = pca(X_rep)
    elif method == "tsne":
        X_2d = tsne(X_rep)
    else:
        raise NotImplementedError

    # plot per label
    for cls in range(5):
        idx = np.argwhere(y_rep == cls)
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=cls,
        )

    plt.legend()
    plt.savefig(os.path.join(config["output_dir"], fname))

    if tb_writer is not None:
        tb_writer.add_figure(f"{method}/{mode}", plt.gcf(), epoch)

    plt.close()
    encoder = encoder.train()
