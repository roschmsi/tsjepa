import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data.dataset import ECGDataset, load_and_split_dataframe
from exp.exp_basic import Exp_Basic
import models.supervised_fedformer.FEDformer as FEDformer
from models.supervised_transformer.run import start_log, write_log
from models.supervised_fedformer.utils import EarlyStopping
from models.supervised_fedformer.metrics import metric
from pathlib import Path
from torch.utils.data import DataLoader

from physionet_evaluation.evaluate_12ECG_score import compute_auc


warnings.filterwarnings("ignore")


class ECGExperiment(Exp_Basic):
    def __init__(self, args):
        super(ECGExperiment, self).__init__(args)

        train_df, val_df, test_df = load_and_split_dataframe(debug=False)
        window = 10 * 500
        src_path = Path("/usr/stud/roschman/datasets/physionet_2020")
        self.train_data = ECGDataset(train_df, window, num_windows=1, src_path=src_path)
        self.val_data = ECGDataset(val_df, window, num_windows=1, src_path=src_path)
        self.test_data = ECGDataset(test_df, window, num_windows=1, src_path=src_path)

    def _build_model(self):
        model = FEDformer.FEDformer(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == "train":
            dataset = self.train_data
        elif flag == "val":
            dataset = self.val_data
        elif flag == "test":
            dataset = self.test_data
        else:
            raise ValueError("wrong flag")

        if flag == "test":
            shuffle_flag = False
        else:
            shuffle_flag = True

        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
        )
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        lbls = []
        probs = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)

                prob = outputs.sigmoid().data.cpu().numpy()
                probs.append(prob)
                lbls.append(batch_y.data.cpu().numpy())

        total_loss = np.average(total_loss)

        lbls = np.concatenate(lbls)
        probs = np.concatenate(probs)
        auroc, auprc = compute_auc(lbls, probs)

        self.model.train()
        return total_loss, auroc

    def train(self, output_dir):
        train_loader = self._get_data(flag="train")
        vali_loader = self._get_data(flag="val")
        test_loader = self._get_data(flag="test")

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        start_log(output_dir)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            lbls = []
            probs = []

            self.model.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                model_optim.zero_grad()

                # encoder - decoder
                outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                prob = outputs.sigmoid().data.cpu().numpy()
                probs.append(prob)
                lbls.append(batch_y.data.cpu().numpy())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            lbls = np.concatenate(lbls)
            probs = np.concatenate(probs)
            train_auroc, train_auprc = compute_auc(lbls, probs)

            vali_loss, vali_auroc = self.vali(vali_loader, criterion)
            test_loss, test_auroc = self.vali(test_loader, criterion)

            write_log(output_dir, epoch, train_loss, train_auroc, vali_loss, vali_auroc)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Vali AUROC: {5:.7f} Test AUROC: {6:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss,
                    vali_auroc,
                    test_auroc,
                )
            )
            early_stopping(vali_loss, self.model, output_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = output_dir / "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, output_dir, test=0):
        test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(output_dir / "checkpoint.pth"))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = output_dir / "results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        f = open("result.txt", "a")
        f.write(str(folder_path) + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path / "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path / "pred.npy", preds)
        np.save(folder_path / "true.npy", trues)

        return

    def predict(self, setting, load=False):
        pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + "real_prediction.npy", preds)

        return
