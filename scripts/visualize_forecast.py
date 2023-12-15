# Reference: https://github.com/gzerveas/mvts_transformer

import logging
import os
import sys
import time
from functools import partial
from runner.forecasting import ForecastingRunner

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import load_dataset
from models.ts_jepa.setup import (
    init_optimizer,
    init_scheduler,
)
from options import Options
from runner.classification import ClassificationRunner
from utils_old import log_training, readable_time, seed_everything, setup
from models.patch_tst.layers.revin import RevIN

from models.ts2vec.ts2vec import TS2VecForecaster, TS2VecClassifier
from models.ts2vec.encoder import TransformerEncoder
from models.ts2vec.utils import (
    load_encoder_from_tsjepa,
    save_checkpoint,
    load_checkpoint,
)
from data.dataset import SupervisedDataset
from data.dataset import create_patch
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")


def denorm(y_pred, revin):
    bs, pred_len, n_vars = y_pred.shape
    y_pred = y_pred.transpose(1, 2).reshape(bs * n_vars, pred_len).unsqueeze(-1)
    y_pred = revin(y_pred, "denorm")
    y_pred = y_pred.squeeze(-1).reshape(bs, n_vars, pred_len).transpose(1, 2)

    return y_pred


def create_model(path_pretrained, config, device, max_seq_len, num_patch):
    encoder = TransformerEncoder(
        seq_len=max_seq_len,
        patch_size=config.patch_len,
        num_patch=num_patch,
        in_chans=config.feat_dim,
        embed_dim=config.enc_d_model,
        depth=config.enc_num_layers,
        num_heads=config.enc_num_heads,
        mlp_ratio=config.enc_mlp_ratio,
        drop_rate=config.dropout,
        attn_drop_rate=config.attn_drop_rate,
        activation=config.activation,
        activation_drop_rate=config.activation_drop_rate,
        norm=config.norm,
        layer_norm_first=config.layer_norm_first,
        learn_pe=config.learn_pe,
        cls_token=config.cls_token,
    )
    model = TS2VecForecaster(
        encoder=encoder,
        n_vars=config.feat_dim,
        d_model=config.enc_d_model,
        num_patch=num_patch,
        forecast_len=config.pred_len,
        patch_len=config.patch_len,
        head_dropout=config.head_dropout,
        head_type=config.head_type,
    )

    model = model.to(device)

    path = os.path.join(path_pretrained, "checkpoints", "model_best.pth")
    model, _, _, epoch = load_checkpoint(
        path,
        model=model,
    )

    model.eval()

    return model


def main(config):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_epoch_time = 0
    total_start_time = time.time()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))

    # config for debugging on single sample
    if config.debug:
        config.batch_size = 2
        config.val_interval = 10
        config.plot_interval = 10
        config.augment = False
        config.dropout = 0

    # build data
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    if config.seq_len is not None:
        max_seq_len = config.seq_len
    else:
        max_seq_len = config.window * config.fs

    if config.use_patch:
        num_patch = (max_seq_len - config.patch_len) // config.stride + 1

    # initialize data generator and runner
    # channel independence, TODO solve for forecasting all dataset
    dataset_class = SupervisedDataset

    # prepare dataloader
    test_dataset = dataset_class(test_dataset)
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     # collate_fn=mask_collator,
    # )

    # ipe = len(train_loader)

    # create model
    model_tsjepa = create_model(
        path_pretrained=config.tsjepa,
        config=config,
        device=device,
        max_seq_len=max_seq_len,
        num_patch=num_patch,
    )
    model_supervised = create_model(
        path_pretrained=config.supervised,
        config=config,
        device=device,
        max_seq_len=max_seq_len,
        num_patch=num_patch,
    )

    if config.revin:
        revin = RevIN(
            num_features=1,
            affine=config.revin_affine,
            subtract_last=False,
        )
        revin = revin.to(device)
    else:
        revin = None

    for ts_id in [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
        X, y = test_dataset[ts_id]
        X = X.to(device)

        X = X.unsqueeze(0)

        # X: (bs x seq_len x n_vars)
        bs, seq_len, n_vars = X.shape
        X = X.transpose(1, 2).reshape(bs * n_vars, seq_len).unsqueeze(-1)
        X = revin(X, "norm")
        X = X.squeeze(-1).reshape(bs, n_vars, seq_len).transpose(1, 2)

        # create patch
        X = create_patch(X, patch_len=config.patch_len, stride=config.stride)

        y_pred_tsjepa = model_tsjepa(X)
        y_pred_supervised = model_supervised(X)

        y_pred_tsjepa = denorm(y_pred_tsjepa, revin).squeeze(0).cpu().detach().numpy()
        y_pred_supervised = (
            denorm(y_pred_supervised, revin).squeeze(0).cpu().detach().numpy()
        )

        # plot y_pred
        for i in range(y_pred_tsjepa.shape[-1]):
            plt.figure(figsize=(8, 6))

            plt.plot(y_pred_tsjepa[:, i], label="TS-JEPA fine-tuned")
            plt.plot(y_pred_supervised[:, i], label="PatchTST supervised")
            plt.plot(y[:, i], label="Ground truth")
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(config.output_dir, f"forecast_{ts_id}_{i}.png"))
            plt.close()


if __name__ == "__main__":
    options = Options()
    options.parser.add_argument("--supervised", type=str)
    options.parser.add_argument("--tsjepa", type=str)

    args = options.parse()
    config = setup(args)
    main(config)
