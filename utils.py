import builtins
import json
import logging
import os
import random
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import yaml
from easydict import EasyDict

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def check_config(config):
    dir = ""
    # check for transformer parameters
    if "d_model" in config.model.keys():
        dir += f"_dmodel={config.model.d_model}"
    if "d_ff" in config.model.keys():
        dir += f"_dff={config.model.d_ff}"
    if "num_layers" in config.model.keys():
        dir += f"_nlayers={config.model.num_layers}"
    if "num_heads" in config.model.keys():
        dir += f"_nheads={config.model.num_heads}"
    if "dropout" in config.model.keys():
        dir += f"_dropout={config.model.dropout}"
    if "pos_encoding" in config.model.keys():
        dir += f"_pe={config.model.pos_encoding}"

    # check for pretraining parameters
    if "masking_ratio" in config.model.keys():
        dir += f"_maskratio={config.model.masking_ratio}"
    if "mean_mask_length" in config.model.keys():
        dir += f"_masklen={config.model.mean_mask_length}"

    # check for fedformer parameters
    if "version" in config.model.keys():
        dir += f"_version={config.model.version}"
    if "modes" in config.model.keys():
        dir += f"_modes={config.model.modes}"

    return dir


def create_output_directory(config):
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config["output_dir"]

    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(
                output_dir
            )
        )

    output_dir = os.path.join(output_dir, config.model.name)
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config["initial_timestamp"] = formatted_timestamp
    formatted_model_config = (
        f"_set={config.data.subset}_window={config.data.window}_fs={config.data.fs}"
        f"_bs={config.training.batch_size}_lr={config.training.batch_size}"
    )
    formatted_model_config += check_config(config)

    if not config.description == "":
        config.description = "_" + config.description
    if config.debug:
        config.description = config.description + "_debug"

    output_dir = os.path.join(
        output_dir, formatted_timestamp + formatted_model_config + config.description
    )

    return config, output_dir


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """
    config = args.__dict__  # configuration dictionary
    model_config = load_config_yaml(config["config_path"])
    config.update(model_config)
    data_config = load_config_yaml("data/dataset.yaml")
    config.update(data_config)
    config = EasyDict(config)

    config, output_dir = create_output_directory(config)
    config["output_dir"] = output_dir
    config["checkpoint_dir"] = os.path.join(output_dir, "checkpoints")
    create_dirs([config["checkpoint_dir"]])

    # Save configuration as json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if not (optimizer is None):
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)


def load_model(
    model,
    model_path,
    optimizer=None,
    resume=False,
    change_output=False,
    lr=None,
):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint["state_dict"])
    if change_output:
        for key, val in checkpoint["state_dict"].items():
            if key.startswith("output_layer"):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded model from {}. Epoch: {}".format(model_path, checkpoint["epoch"]))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def log_training(
    epoch,
    aggr_metrics_train,
    tb_writer,
    start_epoch,
    total_epoch_time,
    epoch_start_time,
    epoch_end_time,
    num_batches,
    num_samples,
):
    print()
    epoch_runtime = epoch_end_time - epoch_start_time
    print_str = "Epoch {} Training Summary: ".format(epoch)
    for k, v in aggr_metrics_train.items():
        tb_writer.add_scalar("{}/train".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)
    logger.info(
        "Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(epoch_runtime)
        )
    )

    total_epoch_time += epoch_runtime
    avg_epoch_time = total_epoch_time / (epoch - start_epoch)
    avg_batch_time = avg_epoch_time / num_batches
    avg_sample_time = avg_epoch_time / num_samples

    logger.info(
        "Avg epoch training time: {} hours, {} minutes, {} seconds".format(
            *readable_time(avg_epoch_time)
        )
    )
    logger.info("Avg batch training time: {} seconds".format(avg_batch_time))
    logger.info("Avg sample training time: {} seconds".format(avg_sample_time))


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config_yaml(config_dir):
    with open(config_dir) as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml
