# Reference: https://github.com/gzerveas/mvts_transformer

import builtins
import json
import logging
import os
import random
import sys
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
    # check dataset
    dir = f"{config.dataset}"

    # check prediction parameters
    if config.seq_len is not None:
        dir += f"_sl={config.seq_len}"
    if config.pred_len is not None:
        dir += f"_pl={config.pred_len}"

    # check patch parameters
    if config.use_patch:
        assert config.patch_len is not None
        assert config.stride is not None
        dir += f"_pat={config.patch_len}"
        dir += f"_str={config.stride}"

        if config.masking_ratio > 0:
            dir += f"_mratio={config.masking_ratio}"

    if config.task in ["classification", "finetuning"]:
        assert config.masking_ratio == 0

    # check training parameters
    if config.batch_size is not None:
        dir += f"_bs={config.batch_size}"
    if config.scheduler is not None:
        if config.scheduler == "CosineAnnealingLR":
            dir += "_sch=CA"
        if config.scheduler == "CosineAnnealingLRWithWarmup":
            dir += "_sch=CAW"
    if config.lr is not None:
        dir += f"_lr={config.lr}"
    if config.weight_decay is not None:
        dir += f"_wd={config.weight_decay}"

    if config.freeze:
        dir += f"_fr={config.freeze_epochs}"

    # vic regularization loss weights
    if config.pred_weight is not None:
        dir += f"_pred={config.pred_weight}"
    if config.std_weight is not None:
        dir += f"_std={config.std_weight}"
    if config.cov_weight is not None:
        dir += f"_cov={config.cov_weight}"

    if config.model_name == "tsjepa":
        if config.no_ema:
            dir += "_noema"
        else:
            dir += f"_ema={config.ema_start}"

    # reverse instance normalization
    if config.revin:
        dir += "_revin"

    # check transformer parameters
    if config.enc_num_layers is not None:
        dir += "_enc"
        dir += f"_l={config.enc_num_layers}"
    if config.enc_num_heads is not None:
        dir += f"_h={config.enc_num_heads}"
    if config.enc_d_model is not None:
        dir += f"_d={config.enc_d_model}"
    if config.enc_d_ff is not None:
        dir += f"_ff={config.enc_d_ff}"
    if config.dec_num_layers is not None:
        dir += "_dec"
        dir += f"_l={config.dec_num_layers}"
    if config.dec_num_heads is not None:
        dir += f"_h={config.dec_num_heads}"
    if config.dec_d_model is not None:
        dir += f"_d={config.dec_d_model}"
    if config.dec_d_ff is not None:
        dir += f"_ff={config.dec_d_ff}"

    if config.dropout is not None:
        dir += f"_drop={config.dropout}"
    if config.learn_pe:
        dir += "_learnpe"
    if config.norm is not None:
        dir += f"_norm={config.norm}"

    return dir


def create_output_directory(config):
    # Create output directory
    initial_timestamp = datetime.now()
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config.initial_timestamp = formatted_timestamp

    output_dir = config.output_dir
    if not os.path.isdir(output_dir):
        raise ValueError(f"Root directory {output_dir} for outputs does not exist.")
    if not config.finetuning:
        output_dir = os.path.join(output_dir, config.model_name, config.task)
    else:
        output_dir = os.path.join(
            output_dir, config.model_name, f"{config.task}_finetuning"
        )
    if config.robustness:
        output_dir = os.path.join(output_dir, "robustness")

    config_description = check_config(config)
    if not config.description == "":
        config_description += f"_{config.description}"
    if config.debug:
        config_description = f"debug_{config_description}"
    output_dir = os.path.join(output_dir, config_description)

    config.output_dir = output_dir
    config.checkpoint_dir = os.path.join(output_dir, "checkpoints")

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
     # configuration dictionary
    config = args.__dict__
    data_config = load_config_yaml(config["data_config"])
    config.update(data_config)
    config = EasyDict(config)

    config, output_dir = create_output_directory(config)

    if os.path.exists(output_dir):
        if not config.debug:
            config.resume = True

    create_dirs([config.checkpoint_dir])

    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


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


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


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
