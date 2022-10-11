#!/usr/bin/env python
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal

from evaluate_12ECG_score import compute_auc, compute_beta_measures, load_weights, compute_challenge_metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from supervised_transformer.model import CTN
from dataset import ECGDataset
from supervised_transformer.optimizer import NoamOpt

from utils import *
from pathlib import Path

patience_count = 0
best_auroc = 0.

def train_12ECG_classifier(input_directory, output_directory):
    src_path = Path(input_directory)
    train_classifier(src_path, output_directory, 3)
    # train_classifier(src_path, output_directory, 4)
    # train_classifier(src_path, output_directory, 6)

def train_classifier(src_path, output_directory, tst_fold):
    global patience_count, best_auroc
    patience_count = 0
    best_auroc = 0.

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    # Train, validation and test fold splits
    val_fold = (tst_fold - 1) % 10
    trn_fold = np.delete(np.arange(10), [val_fold, tst_fold])

    print('trn:', trn_fold)
    print('val:', val_fold)
    print('tst:', tst_fold)

    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, classes).to(device)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    print(f'Number of params: {sum([p.data.nelement() for p in model.parameters()])}')

    trn_df = data_df[data_df.fold.isin(trn_fold)]
    val_df = data_df[data_df.fold == val_fold]
    tst_df = data_df[data_df.fold == tst_fold]

    if debug:
        trn_df = trn_df[:5]
        val_df = val_df[:5]
        tst_df = tst_df[:5]

    trnloader = DataLoader(ECGDataset(trn_df, window, nb_windows=1, src_path=src_path), batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(ECGDataset(val_df, window, nb_windows=10, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=0)
    tstloader = DataLoader(ECGDataset(tst_df, window, nb_windows=20, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=0)
   
    optimizer = NoamOpt(d_model, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # Create dir structure and init logs
    results_loc, sw = create_experiment_directory(output_directory)
    fold_loc = create_fold_dir(results_loc, tst_fold)
    start_log(fold_loc, tst_fold)

    print(fold_loc)

    if do_train:
        for epoch in range(100):
            trn_loss, trn_auroc = train(epoch, model, trnloader, optimizer)
            val_loss, val_auroc = validate(epoch, model, valloader, optimizer, fold_loc)
            write_log(fold_loc, tst_fold, epoch, trn_loss, trn_auroc, val_loss, val_auroc)
            print(f'Train - loss: {trn_loss}, auroc: {trn_auroc}')
            print(f'Valid - loss: {val_loss}, auroc: {val_auroc}')
            
            sw.add_scalar(f'{tst_fold}/trn/loss', trn_loss, epoch)
            sw.add_scalar(f'{tst_fold}/trn/auroc', trn_auroc, epoch)
            sw.add_scalar(f'{tst_fold}/val/loss', val_loss, epoch)
            sw.add_scalar(f'{tst_fold}/val/auroc', val_auroc, epoch)

            # Early stopping
            if patience_count >= patience:
                print(f'Early stopping invoked at epoch, #{epoch}')
                break
        
    # Training done, choose threshold...
    model = load_best_model(str(f'{fold_loc}/{model_name}.tar'), model)

    valloader = DataLoader(ECGDataset(val_df, window, nb_windows=30, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=0)
    tstloader = DataLoader(ECGDataset(tst_df, window, nb_windows=30, src_path=src_path), batch_size=batch_size, shuffle=False, num_workers=0)

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
        np.savetxt(str(fold_loc/'thrs.txt'), thrs)
    else:
        thrs = np.loadtxt(str(fold_loc/'thrs.txt'))
        preds = (probs > thrs).astype(np.int)

    print(thrs)

    f_beta_measure, g_beta_measure = compute_beta_measures(lbls, preds, beta)
    geom_mean = np.sqrt(f_beta_measure*g_beta_measure)
    challenge_metric = compute_challenge_metric(load_weights(weights_file, classes), 
                                                lbls, preds, classes, normal_class)

    with open(fold_loc/f'val_{val_fold}_results.csv', 'w') as f:
        f.write(f'Fbeta_measure, Gbeta_measure, geom_mean, challenge_metric\n')
        f.write(f'{f_beta_measure}, {g_beta_measure}, {geom_mean}, {challenge_metric}\n')
        
    print(f'Validation metrics, fold {val_fold}:')
    print('Fbeta_measure:', f_beta_measure)
    print('Gbeta_measure:', g_beta_measure)
    print('Geometric Mean:', geom_mean)
    print('Challenge_metric:', challenge_metric)    

    # Test
    probs, lbls = get_probs(model, tstloader)
    preds = (probs > thrs).astype(np.int)

    f_beta_measure, g_beta_measure = compute_beta_measures(lbls, preds, beta)
    geom_mean = np.sqrt(f_beta_measure*g_beta_measure)
    challenge_metric = compute_challenge_metric(load_weights(weights_file, classes), 
                                                lbls, preds, classes, normal_class)

    with open(fold_loc/f'tst_{tst_fold}_results.csv', 'w') as f:
        f.write(f'Fbeta_measure, Gbeta_measure, geom_mean, challenge_metric\n')
        f.write(f'{f_beta_measure}, {g_beta_measure}, {geom_mean}, {challenge_metric}\n')
        
    print(f'Test metrics, fold {tst_fold}:')
    print('Fbeta_measure:', f_beta_measure)
    print('Gbeta_measure:', g_beta_measure)
    print('Geometric Mean:', geom_mean)
    print('Challenge_metric:', challenge_metric)

def train(epoch, model, trnloader, optimizer):
    model.train()
    losses, probs, lbls = [], [], []
    for i, (inp_t, lbl_t, filename) in tqdm(enumerate(trnloader), total=len(trnloader)):        
        # Train instances use only one window
        inp_t, lbl_t = inp_t.transpose(1, 0)[0].float().to(device), lbl_t.float().to(device)
        
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
    
    for i, (inp_windows_t, lbl_t, filename) in tqdm(enumerate(valloader), total=len(valloader)):
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
    global patience_count, best_auroc
    patience_count += 1
    if auroc > best_auroc:
        best_auroc = auroc
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
    results_loc = Path(output_directory)/'saved_models'
    results_loc.mkdir(exist_ok=True)

    results_loc = results_loc/model_name
    results_loc.mkdir(exist_ok=True)

    sw = SummaryWriter(log_dir=results_loc)
    return results_loc, sw   

def create_fold_dir(results_loc, fold):
    fold_loc = results_loc/f'fold_{fold}'
    fold_loc.mkdir(exist_ok=True)
    return fold_loc

def start_log(loc, fold):
    if not (loc/f'log_fold_{fold}.csv').exists():
        with open(loc/f'log_fold_{fold}.csv', 'w') as f:
            f.write('epoch, trn_loss, trn_auroc, val_loss, val_auroc\n')

def write_log(loc, fold, epoch, trn_loss, trn_auroc, val_loss, val_auroc):
    with open(loc/f'log_fold_{fold}.csv', 'a') as f:
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

    for i, (inp_windows_t, feats_t, lbl_t, hdr, filename) in tqdm(enumerate(dataloader), total=len(dataloader)):
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