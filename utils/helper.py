import logging
import os
import sys

import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


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

        # loading encoder
        pretrained_dict = checkpoint["model"]
        msg = model.load_state_dict(pretrained_dict)
        logger.info(f"Loaded pretrained model from epoch {epoch} with msg: {msg}")

        # if "teacher" in checkpoint.keys():
        #     teacher_dict = checkpoint["teacher"]
        #     msg = model.ema.model.load_state_dict(teacher_dict)
        #     logger.info(f"loaded pretrained teacher from epoch {epoch} with msg: {msg}")
        # else:
        #     msg = model.ema.model.load_state_dict(pretrained_dict)

        # loading optimizer and scheduler
        if optimizer is not None:
            msg = optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded optimizer from epoch {epoch} with msg: {msg}")

        if scheduler is not None:
            msg = scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"Loaded scheduler from epoch {epoch} with msg: {msg}")

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

        # loading encoder
        pretrained_dict = {}

        for k, v in checkpoint["model"].items():
            if k.startswith("encoder"):
                pretrained_dict[k.replace("encoder.", "")] = v

        msg = model.load_state_dict(pretrained_dict)
        logger.info(f"Loaded pretrained model from epoch {epoch} with msg: {msg}")

        # loading optimizer and scheduler
        if optimizer is not None:
            msg = optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded optimizer from epoch {epoch} with msg: {msg}")

        if scheduler is not None:
            msg = scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"Loaded scheduler from epoch {epoch} with msg: {msg}")

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return model, optimizer, scheduler, epoch


def load_encoder_from_tsjepa(path, encoder):
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # loading encoder
        pretrained_dict = checkpoint["model"]

        # remove encoder from keys
        pretrained_dict = {
            k.replace("encoder.", ""): v for k, v in pretrained_dict.items()
        }

        # drop predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if not k.startswith("predictor")
        }
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder
