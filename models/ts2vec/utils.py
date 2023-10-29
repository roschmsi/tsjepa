import os
import torch
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def save_checkpoint(
    epoch,
    model,
    optimizer,
    path,
    better,
):
    save_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
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
):
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["model"]
        msg = model.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained model from epoch {epoch} with msg: {msg}")

        # -- loading optimizer
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["opt"])

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return model, optimizer, epoch


def load_encoder_from_ts2vec(path, encoder):
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
            k: v for k, v in pretrained_dict.items() if not k.startswith("regression")
        }
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained model from epoch {epoch} with msg: {msg}")

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder
