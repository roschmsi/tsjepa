import os
import random
import torch
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from datetime import datetime


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_experiment_directory(output_dir, model_name):
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_timestamp = datetime.now()
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = model_name + "_" + formatted_timestamp

    exp_dir = output_dir / exp_dir
    exp_dir.mkdir(exist_ok=False)

    sw = SummaryWriter(log_dir=exp_dir)
    return exp_dir, sw


def start_log(loc):
    if not (loc / "log.csv").exists():
        with open(loc / "log.csv", "w") as f:
            f.write("epoch, trn_loss, trn_auroc, val_loss, val_auroc\n")


def write_log(loc, epoch, trn_loss, trn_auroc, val_loss, val_auroc):
    with open(loc / "log.csv", "a") as f:
        f.write(f"{epoch}, {trn_loss}, {trn_auroc}, {val_loss}, {val_auroc}\n")


def load_best_model(model_loc, model):
    checkpoint = torch.load(model_loc)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        "Loading best model: best_loss",
        checkpoint["best_loss"],
        "best_auroc",
        checkpoint["best_auroc"],
        "at epoch",
        checkpoint["epoch"],
    )
    return model
