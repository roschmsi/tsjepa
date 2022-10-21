#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
import yaml
from easydict import EasyDict
from models.supervised_transformer.utils import (
    create_experiment_directory,
    load_best_model,
    seed_everything,
    start_log,
    write_log,
)

from physionet_evaluation.evaluate_12ECG_score import (
    compute_auc,
    compute_beta_measures,
    load_weights,
    compute_challenge_metric,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.supervised_transformer.model import CTN
from data.dataset import ECGDataset, load_and_split_dataframe, classes, normal_class
from models.supervised_transformer.optimizer import NoamOpt


def evaluate_metrics(
    lbls, preds, results_loc, split, beta, weights_file, classes, normal_class
):
    f_beta_measure, g_beta_measure = compute_beta_measures(lbls, preds, beta)
    geom_mean = np.sqrt(f_beta_measure * g_beta_measure)
    challenge_metric = compute_challenge_metric(
        load_weights(weights_file, classes), lbls, preds, classes, normal_class
    )

    with open(results_loc / f"{split}_results.csv", "w") as f:
        f.write("Fbeta_measure, Gbeta_measure, geom_mean, challenge_metric\n")
        f.write(
            f"{f_beta_measure}, {g_beta_measure}, {geom_mean}, {challenge_metric}\n"
        )

    print(f"{split} metrics: ")
    print("Fbeta_measure:", f_beta_measure)
    print("Gbeta_measure:", g_beta_measure)
    print("Geometric Mean:", geom_mean)
    print("Challenge_metric:", challenge_metric)


def train_12ECG_classifier(config):
    seed_everything()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train, val, test split
    train_df, val_df, test_df = load_and_split_dataframe(debug=config.training.debug)
    trainloader = DataLoader(
        ECGDataset(
            train_df,
            window=config.data.window,
            nb_windows=config.data.nb_windows,
            src_path=config.data.dir,
            filter_bandwidth=config.data.filter_bandwidth,
            fs=config.data.fs,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
    )
    valloader = DataLoader(
        ECGDataset(
            val_df,
            window=config.data.window,
            nb_windows=1,
            src_path=config.data.dir,
            filter_bandwidth=config.data.filter_bandwidth,
            fs=config.data.fs,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
    )
    testloader = DataLoader(
        ECGDataset(
            test_df,
            window=config.data.window,
            nb_windows=1,
            src_path=config.data.dir,
            filter_bandwidth=config.data.filter_bandwidth,
            fs=config.data.fs,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # load and initialize model
    model = CTN(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        d_ff=config.model.d_ff,
        num_layers=config.model.num_layers,
        num_classes=config.data.num_classes,
    ).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(f"Number of params: {sum([p.data.nelement() for p in model.parameters()])}")

    optimizer = NoamOpt(
        model_size=config.model.d_model,
        factor=1,
        warmup=4000,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        ),
    )

    # Create dir structure and init logs
    results_loc, sw = create_experiment_directory(
        output_dir=config.output_directory, model_name=config.model.name
    )
    start_log(results_loc)

    # train model
    if config.do_train:
        patience_count = 0
        best_val_loss = 1

        for epoch in range(config.training.epochs):
            trn_loss, trn_auroc = train(model, trainloader, optimizer, device=device)
            val_loss, val_auroc = validate(model, valloader, device=device)
            write_log(results_loc, epoch, trn_loss, trn_auroc, val_loss, val_auroc)
            print(f"Epoch: {epoch}")
            print(f"Train - loss: {trn_loss}, auroc: {trn_auroc}")
            print(f"Valid - loss: {val_loss}, auroc: {val_auroc}")

            sw.add_scalar("train/loss", trn_loss, epoch)
            sw.add_scalar("train/auroc", trn_auroc, epoch)
            sw.add_scalar("val/loss", val_loss, epoch)
            sw.add_scalar("val/auroc", val_auroc, epoch)

            # Save model if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "arch": model.__class__.__name__,
                        "optim_state_dict": optimizer.optimizer.state_dict(),
                        "model_state_dict": model.state_dict(),
                        "best_loss": val_loss,
                        "best_auroc": val_auroc,
                    },
                    str(f"{results_loc}/{config.model.name}.tar"),
                )
                with open(results_loc / "results.csv", "w") as f:
                    f.write("best_epoch, loss, auroc\n")
                    f.write("{epoch}, {loss}, {auroc}\n")

            # Early stopping
            if patience_count >= config.training.patience:
                print(f"Early stopping invoked at epoch #{epoch}")
                break

    # training done, choose threshold on validation set
    model = load_best_model(str(f"{results_loc}/{config.model.name}.tar"), model)

    probs, lbls = get_probs(model, valloader, device=device)

    if config.do_train:
        step = 0.02
        scores = []
        w = load_weights(config.evaluation.weights_file, classes)
        for thr in np.arange(0.0, 1.0, step):
            preds = (probs > thr).astype(int)
            challenge_metric = compute_challenge_metric(
                w, lbls, preds, classes, normal_class
            )
            scores.append(challenge_metric)
        scores = np.array(scores)

        # Best thrs and preds
        idxs = np.argmax(scores, axis=0)
        thrs = np.array([idxs * step])
        preds = (probs > thrs).astype(int)

        # Save
        np.savetxt(str(results_loc / "thrs.txt"), thrs)
    else:
        thrs = np.loadtxt(str(results_loc / "thrs.txt"))
        preds = (probs > thrs).astype(int)

    evaluate_metrics(
        lbls,
        preds,
        results_loc,
        split="Val",
        beta=config.evaluation.beta,
        weights_file=config.evaluation.weights_file,
        classes=classes,
        normal_class=normal_class,
    )

    # test model on test set
    probs, lbls = get_probs(model, testloader, device=device)
    preds = (probs > thrs).astype(int)

    evaluate_metrics(
        lbls,
        preds,
        results_loc,
        split="Test",
        beta=config.evaluation.beta,
        weights_file=config.evaluation.weights_file,
        classes=classes,
        normal_class=normal_class,
    )


def train(model, trnloader, optimizer, device):
    model.train()
    losses, probs, lbls = [], [], []
    for i, (inp_t, lbl_t) in tqdm(enumerate(trnloader), total=len(trnloader)):
        # Train instances use only one window
        inp_t, lbl_t = inp_t.transpose(1, 2).float().to(device), lbl_t.float().to(
            device
        )

        # Train network
        optimizer.optimizer.zero_grad()
        out = model(inp_t)
        loss = F.binary_cross_entropy_with_logits(out, lbl_t)
        loss.backward()
        optimizer.step()

        # Collect loss, probs and labels
        prob = out.sigmoid().data.cpu().numpy()
        losses.append(loss.item())
        probs.append(prob)
        lbls.append(lbl_t.data.cpu().numpy())

    # Epoch results
    loss = np.mean(losses)

    # Compute challenge metrics for overall epoch
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    auroc, auprc = compute_auc(lbls, probs)

    return loss, auroc


def validate(model, valloader, device):
    model.eval()
    losses, probs, lbls = [], [], []

    for i, (inp_windows_t, lbl_t) in tqdm(enumerate(valloader), total=len(valloader)):
        # Get normalized data
        inp_windows_t, lbl_t = inp_windows_t.transpose(1, 2).float().to(
            device
        ), lbl_t.float().to(device)

        # Predict
        with torch.no_grad():
            out = model(inp_windows_t)
        loss = F.binary_cross_entropy_with_logits(out, lbl_t)

        # Collect loss, probs and labels
        losses.append(loss.item())
        prob = out.sigmoid().data.cpu().numpy()
        probs.append(prob)
        lbls.append(lbl_t.data.cpu().numpy())

    # Epoch results
    loss = np.mean(losses)

    # Compute challenge metrics for overall epoch
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    auroc, auprc = compute_auc(lbls, probs)

    return loss, auroc


def get_probs(model, dataloader, device):
    """Return probs and lbls given model and dataloader"""
    model.eval()
    probs, lbls = [], []

    for i, (inp_windows_t, lbl_t) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Get normalized data
        inp_windows_t, lbl_t = inp_windows_t.transpose(1, 2).float().to(
            device
        ), lbl_t.float().to(device)

        # Predict
        with torch.no_grad():
            out = model(inp_windows_t)

        # Collect probs and labels
        probs.append(out.sigmoid().data.cpu().numpy())
        lbls.append(lbl_t.data.cpu().numpy())

    # Consolidate probs and labels
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    return probs, lbls


if __name__ == "__main__":
    # argparse config and train or test mode

    with open(
        "/usr/stud/roschman/ECGAnalysis/models/supervised_transformer/supervised_transformer.yaml"
    ) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    print("Running training code...")

    train_12ECG_classifier(config)

    print("Done.")
