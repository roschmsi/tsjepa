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

    # check dataset
    if "dataset" in config.keys():
        dir += f"data={config.dataset}"
    if "augment" in config.keys():
        dir += f"_augment={config.augment}"
    if "mixup" in config.keys():
        dir += f"_mixup={config.mixup}"

    # check patch parameters
    if "patch_len" in config.keys():
        dir += f"_patch={config.patch_len}"
    if "stride" in config.keys():
        dir += f"_stride={config.stride}"
    if "masking_ratio" in config.keys() and config.masking_ratio > 0:
        dir += f"_mratio={config.masking_ratio}"
    if "masking_ratio_pretraining" in config.keys():
        dir += f"_premratio={config.masking_ratio_pretraining}"
    if "mean_mask_length" in config.keys():
        dir += f"_mlen={config.mean_mask_length}"

    # check training parameters
    if "batch_size" in config.keys():
        dir += f"_bs={config.batch_size}"
    if "optimizer" in config.keys():
        dir += f"_opt={config.optimizer}"
    if "scheduler" in config.keys() and config.scheduler != "":
        dir += f"_sched={config.scheduler}"
    if "lr" in config.keys():
        dir += f"_lr={config.lr}"

    # check transformer parameters
    if "d_model" in config.keys():
        dir += f"_dmodel={config.d_model}"
    if "d_ff" in config.keys():
        dir += f"_dff={config.d_ff}"
    if "num_layers" in config.keys():
        dir += f"_nlayers={config.num_layers}"
    if "num_heads" in config.keys():
        dir += f"_nheads={config.num_heads}"
    if "pos_encoding" in config.keys():
        dir += f"_pe={config.pos_encoding}"
    if "num_cnn" in config.keys():
        dir += f"_num_cnn={config.num_cnn}"

    # check fedformer parameters
    if "version" in config.keys():
        dir += f"_version={config.version}"
    if "modes" in config.keys():
        dir += f"_modes={config.modes}"

    return dir


def create_output_directory(config):
    # Create output directory
    initial_timestamp = datetime.now()
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config["initial_timestamp"] = formatted_timestamp

    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        raise ValueError(f"Root directory {output_dir} for outputs must exist.")
    output_dir = os.path.join(output_dir, config.model_name)

    config_description = check_config(config)
    if not config.description == "":
        config_description += f"_{config.description}"
    if config.debug:
        config_description = f"debug_{formatted_timestamp}{config_description}"

    output_dir = os.path.join(output_dir, config_description)

    config["output_dir"] = output_dir
    config["checkpoint_dir"] = os.path.join(output_dir, "checkpoints")

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


def setup_tuning(args):
    config = args.__dict__  # configuration dictionary
    model_config = load_config_yaml(config["config_model"])
    config.update(model_config)
    data_config = load_config_yaml(config["config_data"])  # "data/dataset.yaml"
    config.update(data_config)
    config = EasyDict(config)

    initial_timestamp = datetime.now()
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config["formatted_timestamp"] = formatted_timestamp

    output_path = os.path.join(
        os.getcwd(), "output", config.model_name, formatted_timestamp
    )

    os.makedirs(output_path, exist_ok=True)
    with open(
        os.path.join(output_path, "configuration.json"),
        "w",
    ) as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    return config


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """
    config = args.__dict__  # configuration dictionary
    config = EasyDict(config)

    config, output_dir = create_output_directory(config)

    if os.path.exists(output_dir):
        config["resume"] = True

    create_dirs([config["checkpoint_dir"]])

    # Save configuration as json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    # assure that config of pretrained model matches config of finetuned model
    if args.finetune:
        pretraining_config = load_config_yaml(
            os.path.join(args.load_model, "configuration.json")
        )
        pretraining_config = EasyDict(pretraining_config)

        keys = ["dropout", "patch_len", "stride", "use_patch"]

        if config.model_name == "finetuning_patch_tst":
            keys.extend(
                [
                    "d_model",
                    "d_ff",
                    "num_heads",
                    "num_layers",
                ]
            )
        if config.model_name == "finetuning_masked_autoencoder":
            keys.extend(
                [
                    "enc_d_model",
                    "enc_d_ff",
                    "enc_num_heads",
                    "enc_num_layers",
                ]
            )

        for key in keys:
            assert config.model[key] == pretraining_config.model[key]

        assert config.masking_ratio_pretraining == pretraining_config.masking_ratio

    return config


def save_model(path, epoch, model, optimizer=None):
    data = {"epoch": epoch, "state_dict": model.state_dict()}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)


def load_model(
    model,
    path,
    optimizer=None,
    resume=False,
    change_output=False,
):
    start_epoch = 0
    checkpoint = torch.load(path)
    state_dict = deepcopy(checkpoint["model_state_dict"])

    if change_output:
        for key, _ in checkpoint["model_state_dict"].items():
            if (
                key.startswith("head")
                or key.startswith("decoder")
                or key.startswith("encoder_pos_embed")
            ):
                state_dict.pop(key)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    print("Loaded model from {}. Epoch: {}".format(path, checkpoint["epoch"]))

    if resume:
        start_epoch = checkpoint["epoch"]
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("No optimizer parameters in checkpoint.")
        return model, optimizer, start_epoch

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
