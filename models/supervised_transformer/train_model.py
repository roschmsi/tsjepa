#!/usr/bin/env python
from datetime import datetime
import sys
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal

from physionet_evaluation.evaluate_12ECG_score import compute_auc, compute_beta_measures, load_weights, compute_challenge_metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from models.supervised_transformer.model import CTN
from data.dataset import ECGDataset
from models.supervised_transformer.optimizer import NoamOpt

from utils import *
from pathlib import Path
import random
import os

patience_count = 0
best_auroc = 0.
best_loss = 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def evaluate_metrics(lbls, preds, results_loc, split):
    f_beta_measure, g_beta_measure = compute_beta_measures(lbls, preds, beta)
    geom_mean = np.sqrt(f_beta_measure*g_beta_measure)
    challenge_metric = compute_challenge_metric(load_weights(weights_file, classes), 
                                                lbls, preds, classes, normal_class)

    with open(results_loc/f'{split}_results.csv', 'w') as f:
        f.write(f'Fbeta_measure, Gbeta_measure, geom_mean, challenge_metric\n')
        f.write(f'{f_beta_measure}, {g_beta_measure}, {geom_mean}, {challenge_metric}\n')
        
    print(f'{split} metrics: ')
    print('Fbeta_measure:', f_beta_measure)
    print('Gbeta_measure:', g_beta_measure)
    print('Geometric Mean:', geom_mean)
    print('Challenge_metric:', challenge_metric)

def train_12ECG_classifier(input_directory, output_directory):
    src_path = Path(input_directory)
    seed_everything()

    global patience_count, best_auroc, best_loss
    patience_count = 0
    best_auroc = 0.
    best_loss= 1

    # train, val, test split
    data_df = pd.read_csv('data/records_stratified_10_folds_v2.csv', index_col=0).reset_index(drop=True)
    # filter for ptb-xl data
    data_df = data_df[data_df['Patient'].str.contains('HR')].reset_index(drop=True)

    train_df = data_df.sample(frac=0.8,random_state=42)
    data_df = data_df.drop(train_df.index)
    val_df= data_df.sample(frac=0.5, random_state=42)
    test_df = data_df.drop(val_df.index)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if debug:
        train_df = train_df[:2]
        val_df = train_df[:2]
        test_df = train_df[:2]

    trnloader = DataLoader(ECGDataset(train_df, window, nb_windows=1, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=2)
    valloader = DataLoader(ECGDataset(val_df, window, nb_windows=1, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=2)
    tstloader = DataLoader(ECGDataset(test_df, window, nb_windows=1, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=2)

    # load and initialize model
    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, classes).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(f'Number of params: {sum([p.data.nelement() for p in model.parameters()])}')

    optimizer = NoamOpt(d_model, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # Create dir structure and init logs
    results_loc, sw = create_experiment_directory(output_directory)
    start_log(results_loc)

    # train model
    if do_train:
        for epoch in range(epochs):
            trn_loss, trn_auroc = train(model, trnloader, optimizer)
            val_loss, val_auroc = validate(epoch, model, valloader, optimizer, results_loc)
            write_log(results_loc, epoch, trn_loss, trn_auroc, val_loss, val_auroc)
            print(f'Train - loss: {trn_loss}, auroc: {trn_auroc}')
            print(f'Valid - loss: {val_loss}, auroc: {val_auroc}')
            
            sw.add_scalar(f'trn/loss', trn_loss, epoch)
            sw.add_scalar(f'trn/auroc', trn_auroc, epoch)
            sw.add_scalar(f'val/loss', val_loss, epoch)
            sw.add_scalar(f'val/auroc', val_auroc, epoch)

            # Early stopping
            if patience_count >= patience:
                print(f'Early stopping invoked at epoch #{epoch}')
                break
        
    # training done, choose threshold on validation set
    model = load_best_model(str(f'{results_loc}/{model_name}.tar'), model)

    valloader = DataLoader(ECGDataset(val_df, window, nb_windows=30, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=2)
    tstloader = DataLoader(ECGDataset(test_df, window, nb_windows=30, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=2)

    probs, lbls = get_probs(model, valloader)

    if do_train:
        step = 0.02
        scores = []
        w = load_weights(weights_file, classes)
        for thr in np.arange(0., 1., step):
            preds = (probs > thr).astype(np.int)
            challenge_metric = compute_challenge_metric(w, lbls, preds, classes, normal_class)
            scores.append(challenge_metric)
        scores = np.array(scores)
            
        # Best thrs and preds
        idxs = np.argmax(scores, axis=0)
        thrs = np.array([idxs*step])
        preds = (probs > thrs).astype(np.int)

        # Save
        np.savetxt(str(results_loc/'thrs.txt'), thrs)
    else:
        thrs = np.loadtxt(str(results_loc/'thrs.txt'))
        preds = (probs > thrs).astype(np.int)

    print(thrs)

    evaluate_metrics(lbls, preds, results_loc, split='Val')

    # test model on test set
    probs, lbls = get_probs(model, tstloader)
    preds = (probs > thrs).astype(np.int)

    evaluate_metrics(lbls, preds, results_loc, split='Test')

def train(model, trnloader, optimizer):
    model.train()
    losses, probs, lbls = [], [], []
    for i, (inp_t, lbl_t) in tqdm(enumerate(trnloader), total=len(trnloader)):        
        # Train instances use only one window
        inp_t, lbl_t = inp_t.transpose(1, 2).float().to(device), lbl_t.float().to(device)
        
        # Train network
        optimizer.optimizer.zero_grad()
        out = model(inp_t)
        if class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(out, lbl_t, class_weights)
        else:
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

def validate(epoch, model, valloader, optimizer, fold_loc):
    model.eval()
    losses, probs, lbls = [], [], []
    
    for i, (inp_windows_t, lbl_t) in tqdm(enumerate(valloader), total=len(valloader)):
        # Get normalized data
        inp_windows_t, lbl_t = inp_windows_t.transpose(1, 2).float().to(device), lbl_t.float().to(device)
    
        # Predict
        # outs = []
        with torch.no_grad():
            # Loop over nb_windows
            # for inp_t in inp_windows_t.transpose(1, 0):
            #     out = model(inp_t)
            #     outs.append(out)
            # out = torch.stack(outs).mean(dim=0)   # take the average of the sequence windows
            out = model(inp_windows_t)
        
        if class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(out, lbl_t, class_weights)
        else:
            loss = F.binary_cross_entropy_with_logits(out, lbl_t)

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
    
    # Save model if best
    global patience_count, best_auroc, best_loss
    patience_count += 1
    # if auroc > best_auroc:
    #     best_auroc = auroc
    if loss < best_loss:
        best_loss = loss
        patience_count = 0
        torch.save({'epoch': epoch,
                    'arch': model.__class__.__name__,
                    'optim_state_dict': optimizer.optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'best_loss': loss,
                    'best_auroc' : auroc}, str(f'{fold_loc}/{model_name}.tar'))
        with open(fold_loc/'results.csv', 'w') as f:
            f.write(f'best_epoch, loss, auroc\n')
            f.write(f'{epoch}, {loss}, {auroc}\n')
    
    #lr_sched.step(loss)
    return loss, auroc

def create_experiment_directory(output_directory):
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    initial_timestamp = datetime.now()
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = model_name + "_" + formatted_timestamp

    output_dir = output_dir / dir_name
    output_dir.mkdir(exist_ok=True)

    sw = SummaryWriter(log_dir=output_dir)
    return output_dir, sw

def start_log(loc):
    if not (loc/f'log.csv').exists():
        with open(loc/f'log.csv', 'w') as f:
            f.write('epoch, trn_loss, trn_auroc, val_loss, val_auroc\n')

def write_log(loc, epoch, trn_loss, trn_auroc, val_loss, val_auroc):
    with open(loc/f'log.csv', 'a') as f:
        f.write(f'{epoch}, {trn_loss}, {trn_auroc}, {val_loss}, {val_auroc}\n')                    

def load_best_model(model_loc, model):
    checkpoint = torch.load(model_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loading best model: best_loss', checkpoint['best_loss'], 'best_auroc', checkpoint['best_auroc'], 'at epoch', checkpoint['epoch'])
    return model

def get_probs(model, dataloader):
    ''' Return probs and lbls given model and dataloader '''
    model.eval()
    probs, lbls = [], []

    for i, (inp_windows_t, lbl_t) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Get normalized data
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.float().to(device)

        # Predict
        outs = []
        with torch.no_grad():
            # Loop over nb_windows
            for inp_t in inp_windows_t.transpose(1, 0):
                out = model(inp_t)
                outs.append(out)
            out = torch.stack(outs).mean(dim=0)   # take the average of the sequence windows

        # Collect probs and labels
        probs.append(out.sigmoid().data.cpu().numpy())
        lbls.append(lbl_t.data.cpu().numpy())

    # Consolidate probs and labels
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    return probs, lbls


if __name__ == '__main__':
    # Parse arguments.
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print('Running training code...')

    train_12ECG_classifier(input_directory, output_directory)

    print('Done.')