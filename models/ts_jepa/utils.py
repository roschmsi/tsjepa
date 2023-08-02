import copy
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

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


# function to load the checkpoint for the classifier
def load_classifier_checkpoint(
    path,
    classifier,
    optimizer,
):
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading classifier
        pretrained_dict = checkpoint["classifier"]
        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

        # -- loading optimizer
        optimizer.load_state_dict(checkpoint["opt"])

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return classifier, optimizer, epoch


# load weights for encoder and target encoder from classifier
def load_encoder_from_classifier(
    path,
    classifier,
):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    # -- loading classifier
    pretrained_dict = checkpoint["classifier"]
    msg = classifier.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    encoder = copy.deepcopy(classifier.encoder)
    target_encoder = copy.deepcopy(classifier.encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    return encoder, target_encoder


def load_encoder_from_tsjepa(path, encoder):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    # -- loading classifier
    pretrained_dict = checkpoint["encoder"]
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    return encoder


def save_classifier_checkpoint(
    epoch,
    classifier,
    optimizer,
    path,
    better,
):
    save_dict = {
        "epoch": epoch,
        "classifier": classifier.state_dict(),
        "opt": optimizer.state_dict(),
        # "lr": lr,
        # "loss": loss,
        # "batch_size": batch_size,
    }

    torch.save(save_dict, os.path.join(path, "model_last.pth"))
    if better:
        torch.save(save_dict, os.path.join(path, "model_best.pth"))


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
    num_classes,
    tb_writer=None,
    mode=None,
    epoch=None,
    supervised=False,
):
    encoder = encoder.eval()
    X_rep = []
    y_rep = []

    with torch.no_grad():
        # change dataset class, mean across time axis and
        if supervised:
            for X, y, _ in data_loader:
                # TODO discuss: first mean across channels, then mean across time
                X = X.float().to(device)
                X_enc = encoder(X)
                X_rep.append(X_enc.mean(dim=1).mean(dim=2))
                y_rep.append(y)
        else:
            for X, y, masks_enc, masks_pred in data_loader:
                X = X.float().to(device)
                X_enc = encoder(X, masks=None)
                X_rep.append(X_enc.mean(dim=1))
                y_rep.append(y)
                # y_rep.append(torch.where(y == 1)[1])

        X_rep = torch.cat(X_rep, dim=0).cpu().detach().numpy()
        y_rep = torch.cat(y_rep, dim=0).numpy()

    if method == "pca":
        X_2d = pca(X_rep)
    elif method == "tsne":
        X_2d = tsne(X_rep)
    else:
        raise NotImplementedError

    # plot per label
    for cls in range(num_classes):
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


def plot_classwise_distribution(
    encoder,
    data_loader,
    device,
    d_model,
    num_classes,
    tb_writer=None,
    mode=None,
    epoch=None,
    supervised=False,
):
    encoder = encoder.eval()
    df = None

    with torch.no_grad():
        # change dataset class, mean across time axis and
        if supervised:
            for X, y, _ in data_loader:
                X = X.float().to(device)
                X_enc = encoder(X)
                X_enc = X_enc.mean(dim=1).mean(dim=2)

                df_z = pd.DataFrame(X_enc.cpu()).astype("float")
                df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")

                df_batch = pd.concat([df_y, df_z], axis=1)
                df = pd.concat([df, df_batch], axis=0)
        else:
            for X, y, masks_enc, masks_pred in data_loader:
                X = X.float().to(device)
                X_enc = encoder(X, masks=None)
                X_enc = X_enc.mean(dim=1)

                df_z = pd.DataFrame(X_enc.cpu()).astype("float")
                df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")

                df_batch = pd.concat([df_y, df_z], axis=1)
                df = pd.concat([df, df_batch], axis=0)

    # plot per label
    for dim in range(d_model):
        plt.figure(figsize=(8, 4))

        for cls in range(num_classes):
            # Plot KDE plot for each class
            fig = sns.kdeplot(
                x=dim,
                data=df[df["y"] == cls].reset_index(),
                label=cls,
            )
            fig.set(xlabel=None)
            fig.set(ylabel=None)

        plt.legend()
        # plt.savefig(path / f"kde_dim={dim}.jpg", bbox_inches="tight")
        plt.tight_layout()

        if tb_writer is not None:
            tb_writer.add_figure(
                f"classwise distribution {dim}/{mode}", plt.gcf(), epoch
            )

        plt.close()

    encoder = encoder.train()
