# Reference: https://github.com/gzerveas/mvts_transformer

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
    # check dataset
    dir = f"dataset={config.dataset}"
    # if "fs" in config.keys():
    #     dir += f"_fs={config.fs}"
    if config.filter_bandwidth:
        dir += "_fb"
    if config.augment:
        dir += "_aug"
    if config.mixup is not None:
        dir += f"_mixup={config.mixup}"
    if config.rand_ecg != "":
        dir += f"_rand={config.rand_ecg}"

    # check prediction parameters
    if config.seq_len is not None:
        dir += f"_sl={config.seq_len}"
    if config.pred_len is not None:
        dir += f"_pl={config.pred_len}"

    # check patch parameters
    if config.use_patch:
        assert config.patch_len is not None
        assert config.stride is not None
        dir += f"_patch={config.patch_len}"
        dir += f"_stride={config.stride}"

        if config.masking_ratio > 0:
            dir += f"_mratio={config.masking_ratio}"
        if "masking_ratio_pretraining" in config.keys():
            dir += f"_mratiopre={config.masking_ratio_pretraining}"

    if config.task in ["classification", "finetuning"]:
        assert config.masking_ratio == 0

    if config.mean_mask_length is not None:
        dir += f"_mmlen={config.mean_mask_length}"

    # check training parameters
    if config.batch_size is not None:
        dir += f"_bs={config.batch_size}"
    if config.optimizer is not None:
        dir += f"_opt={config.optimizer}"
    if config.scheduler is not None:
        dir += f"_sch={config.scheduler}"
    if config.lr is not None:
        dir += f"_lr={config.lr}"
    if config.weight_decay is not None:
        dir += f"_wd={config.weight_decay}"

    if config.freeze:
        dir += f"_fr={config.freeze_epochs}"

    # vic regularization loss weights
    if config.vic_reg:
        dir += f"_pred={config.pred_weight}"
        dir += f"_std={config.std_weight}"
        dir += f"_cov={config.cov_weight}"

    # reverse instance normalization
    if config.revin:
        dir += "_revin"

    # hierarchical
    if config.num_levels is not None:
        dir += f"_nlevels={config.num_levels}"
    if config.ch_factor is not None:
        dir += f"_chf={config.ch_factor}"

    if config.window_size is not None:
        dir += f"_wsize={config.window_size}"

    # check transformer parameters
    if config.mae or config.model_name == "tsjepa":
        dir += "_enc"
        if config.enc_num_layers is not None:
            dir += f"_nlayers={config.enc_num_layers}"
        if config.enc_num_heads is not None:
            dir += f"_nheads={config.enc_num_heads}"
        if config.enc_d_model is not None:
            dir += f"_dmodel={config.enc_d_model}"
        if config.enc_d_ff is not None:
            dir += f"_dff={config.enc_d_ff}"
        dir += "_dec"
        if config.dec_num_layers is not None:
            dir += f"_nlayers={config.dec_num_layers}"
        if config.dec_num_heads is not None:
            dir += f"_nheads={config.dec_num_heads}"
        if config.dec_d_model is not None:
            dir += f"_dmodel={config.dec_d_model}"
        if config.dec_d_ff is not None:
            dir += f"_dff={config.dec_d_ff}"
    else:
        if config.num_layers is not None:
            dir += f"_nlayers={config.num_layers}"
        if config.num_heads is not None:
            dir += f"_nheads={config.num_heads}"
        if config.d_model is not None:
            dir += f"_dmodel={config.d_model}"
        if config.d_ff is not None:
            dir += f"_dff={config.d_ff}"

    if config.dropout is not None:
        dir += f"_drop={config.dropout}"
    if config.learn_pe:
        dir += "_learnpe"
    if config.norm is not None:
        dir += f"_norm={config.norm}"
    if config.activation is not None:
        dir += f"_act={config.activation}"

    if config.num_cnn is not None:
        dir += f"_numcnn={config.num_cnn}"
    if config.ch_token:
        dir += "_chtoken"
    if config.cls_token:
        dir += "_clstoken"

    # check tsjepa parameters
    if config.model_name == "tsjepa":
        dir += f"_emastart={config.ema_start}"

    # check fedformer parameters
    if config.version is not None:
        dir += f"_vers={config.version}"
    if config.modes is not None:
        dir += f"_modes={config.modes}"

    return dir


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def create_output_directory(config):
    # Create output directory
    initial_timestamp = datetime.now()
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config["initial_timestamp"] = formatted_timestamp

    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        raise ValueError(f"Root directory {output_dir} for outputs does not exist.")
    if not config.finetuning:
        output_dir = os.path.join(output_dir, config.model_name, config.task)
    else:
        output_dir = os.path.join(
            output_dir, config.model_name, f"{config.task}_finetuning"
        )

    config_description = check_config(config)
    if not config.description == "":
        config_description += f"_{config.description}"
    if config.debug:
        config_description = f"debug_{config_description}"
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
    data_config = load_config_yaml(config["config_data"])
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
    data_config = load_config_yaml(config["data_config"])
    config.update(data_config)
    config = EasyDict(config)

    # assure that config of pretrained model matches config of finetuned model
    if config.finetuning:
        pretraining_config = load_config_yaml(
            os.path.join(args.load_model, "configuration.json")
        )
        pretraining_config = EasyDict(pretraining_config)

        keys = ["dropout"]

        if config.use_patch:
            keys.extend(["use_patch", "patch_len", "stride"])
        if config.mae:
            keys.extend(["enc_d_model", "enc_d_ff", "enc_num_heads", "enc_num_layers"])
        else:
            keys.extend(["d_model", "d_ff", "num_heads", "num_layers"])

        for key in keys:
            assert config[key] == pretraining_config[key]

        config.masking_ratio_pretraining = pretraining_config.masking_ratio

    config, output_dir = create_output_directory(config)

    if os.path.exists(output_dir):
        if not config.debug:
            config["resume"] = True

    create_dirs([config["checkpoint_dir"]])

    # save configuration as json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def save_model(path, epoch, model, optimizer=None):
    data = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(data, path)


def load_model(
    model, path, optimizer=None, resume=False, change_output=False, device=None
):
    start_epoch = 0
    checkpoint = torch.load(path)
    model_state_dict = deepcopy(checkpoint["model_state_dict"])

    if change_output:
        for key, _ in checkpoint["model_state_dict"].items():
            if key.startswith("head") or key.startswith("decoder"):
                model_state_dict.pop(key)

    missing_keys, unexpected_keys = model.load_state_dict(
        model_state_dict, strict=False
    )

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    print("Loaded model from {}. Epoch: {}".format(path, checkpoint["epoch"]))

    if resume:
        start_epoch = checkpoint["epoch"]
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_to(optimizer, device=device)
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
