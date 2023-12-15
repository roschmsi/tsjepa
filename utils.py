# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
from data.dataset import create_patch

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import logging
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    path,
    better,
):
    save_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        # "teacher": model.ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    torch.save(save_dict, os.path.join(path, "model_last.pth"))
    # torch.save(save_dict, os.path.join(path, f"model_{epoch}.pth"))
    if better:
        torch.save(save_dict, os.path.join(path, "model_best.pth"))

    if epoch % 50 == 0:
        torch.save(save_dict, os.path.join(path, f"model_{epoch}.pth"))


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
):
    try:
        checkpoint = torch.load(path)
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["model"]
        msg = model.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained model from epoch {epoch} with msg: {msg}")

        # if "teacher" in checkpoint.keys():
        #     teacher_dict = checkpoint["teacher"]
        #     msg = model.ema.model.load_state_dict(teacher_dict)
        #     logger.info(f"loaded pretrained teacher from epoch {epoch} with msg: {msg}")
        # else:
        #     msg = model.ema.model.load_state_dict(pretrained_dict)

        # -- loading optimizer
        if optimizer is not None:
            msg = optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"loaded optimizer from epoch {epoch} with msg: {msg}")
            # optimizer_to(optimizer, device=device)

        if scheduler is not None:
            msg = scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"loaded scheduler from epoch {epoch} with msg: {msg}")

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return model, optimizer, scheduler, epoch


def load_checkpoint_encoder(
    path,
    model,
    optimizer=None,
    scheduler=None,
):
    try:
        checkpoint = torch.load(path)
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = {}

        for k, v in checkpoint["model"].items():
            if k.startswith("encoder"):
                pretrained_dict[k.replace("encoder.", "")] = v

        msg = model.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained model from epoch {epoch} with msg: {msg}")

        # -- loading optimizer
        if optimizer is not None:
            msg = optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"loaded optimizer from epoch {epoch} with msg: {msg}")
            # optimizer_to(optimizer, device=device)

        if scheduler is not None:
            msg = scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"loaded scheduler from epoch {epoch} with msg: {msg}")

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return model, optimizer, scheduler, epoch


def load_encoder_from_tsjepa(path, encoder):
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["model"]
        # remove encoder from keys
        pretrained_dict = {
            k.replace("encoder.", ""): v for k, v in pretrained_dict.items()
        }
        # drop k, v if k starts with "regression head"
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if not k.startswith("predictor")
        }
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained model from epoch {epoch} with msg: {msg}")

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder


def plot_classwise_distribution(
    encoder,
    data_loader,
    device,
    d_model,
    num_classes,
    revin=None,
    tb_writer=None,
    mode=None,
    epoch=None,
    supervised=False,
    model=None,
    patch_len=0,
    stride=0,
):
    encoder.eval()
    df = None

    with torch.no_grad():
        # change dataset class, mean across time axis and
        if supervised and model == "patch_tst":
            for X, _, _, _, _, _, _ in data_loader:
                if df is not None and df.shape[0] > 100000:
                    break

                X = X.float().to(device)
                X_enc = encoder(X)
                X_enc = X_enc.mean(dim=1).mean(dim=2)

                df_z = pd.DataFrame(X_enc.cpu()).astype("float")

                if num_classes == 1:
                    df_y = pd.DataFrame(
                        torch.zeros(X_enc.shape[0]), columns=["y"]
                    ).astype("int")
                else:
                    df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")

                df_batch = pd.concat([df_y, df_z], axis=1)
                df = pd.concat([df, df_batch], axis=0)
        elif supervised:
            for X, y, _ in data_loader:
                if df is not None and df.shape[0] > 100000:
                    break

                X = X.float().to(device)
                X_enc = encoder(X)
                X_enc = X_enc.mean(dim=1).mean(dim=2)

                df_z = pd.DataFrame(X_enc.cpu()).astype("float")

                if num_classes == 1:
                    df_y = pd.DataFrame(torch.zeros(y.shape[0]), columns=["y"]).astype(
                        "int"
                    )
                else:
                    df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")

                df_batch = pd.concat([df_y, df_z], axis=1)
                df = pd.concat([df, df_batch], axis=0)

        elif model == "ts2vec":
            for X in data_loader:
                if df is not None and df.shape[0] > 100000:
                    break

                X = X.to(device)
                if revin is not None:
                    X = revin(X, "norm")

                X = create_patch(X, patch_len=patch_len, stride=stride)
                X = X.squeeze()
                X_enc = encoder(X)["encoder_out"]
                bs, seq_len, d_model = X_enc.shape
                X_enc = X_enc.reshape(bs * seq_len, d_model)
                # X_enc = X_enc.mean(dim=1)

                df_z = pd.DataFrame(X_enc.cpu()).astype("float")

                if num_classes == 1:
                    df_y = pd.DataFrame(torch.zeros(X.shape[0]), columns=["y"]).astype(
                        "int"
                    )
                else:
                    df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")

                df_batch = pd.concat([df_y, df_z], axis=1)
                df = pd.concat([df, df_batch], axis=0)
        else:
            for X, y, masks_enc, masks_pred in data_loader:
                # if df is not None and df.shape[0] > 100000:
                #     break

                X = X.float().to(device)
                X_enc = encoder(X, masks=None)
                X_enc = X_enc.mean(dim=1)

                df_z = pd.DataFrame(X_enc.cpu()).astype("float")

                if num_classes == 1:
                    df_y = pd.DataFrame(torch.zeros(y.shape[0]), columns=["y"]).astype(
                        "int"
                    )
                else:
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

        # TODO set xlim again
        # plt.xlim([-2.0, 2.0])
        plt.legend()
        # plt.savefig(path / f"kde_dim={dim}.jpg", bbox_inches="tight")
        plt.tight_layout()

        if tb_writer is not None:
            tb_writer.add_figure(
                f"classwise distribution {dim}/{mode}", plt.gcf(), epoch
            )

        plt.close()

    encoder.train()
