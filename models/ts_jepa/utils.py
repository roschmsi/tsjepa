import torch
import logging
import sys
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np

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


def plot_tsne(encoder, data_loader, device, config, fname):
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

    tsne = TSNE(n_components=2, perplexity=2)
    X_2d = tsne.fit_transform(X_rep)

    # plot per label
    # for label in labels:
    #     indices = np.argwhere(y_train == label)
    #     plt.scatter(x_train[indices, 0], x_train[indices, 1], marker='o', s=25,
    #                 c=map_color(label), label=label, edgecolor='k')

    plt.scatter(X_2d[:, 0], X_2d[:, 1], label=np.unique(y_rep), c=y_rep)
    plt.legend()
    plt.savefig(os.path.join(config["output_dir"], fname))
    plt.close()

    encoder = encoder.train()
