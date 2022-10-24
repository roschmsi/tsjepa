import argparse
import torch


class Options(object):
    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description="Autoformer & Transformer family for Time Series Forecasting"
        )

        # basic config
        self.parser.add_argument("--is_training", type=int, default=1, help="status")
        self.parser.add_argument("--task_id", type=str, default="test", help="task id")
        self.parser.add_argument(
            "--model",
            type=str,
            default="FEDformer",
            help="model name, options: [FEDformer, Autoformer, Informer, Transformer]",
        )

        # supplementary config for FEDformer model
        self.parser.add_argument(
            "--version",
            type=str,
            default="Fourier",
            help="for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]",
        )
        self.parser.add_argument(
            "--mode_select",
            type=str,
            default="random",
            help="for FEDformer, there are two mode selection method, options: [random, low]",
        )
        self.parser.add_argument(
            "--modes", type=int, default=64, help="modes to be selected random 64"
        )
        self.parser.add_argument("--L", type=int, default=3, help="ignore level")
        self.parser.add_argument(
            "--base", type=str, default="legendre", help="mwt base"
        )
        self.parser.add_argument(
            "--cross_activation",
            type=str,
            default="tanh",
            help="mwt cross atention activation function tanh or softmax",
        )

        # data loader
        self.parser.add_argument(
            "--data", type=str, default="ETTh1", help="dataset type"
        )
        self.parser.add_argument(
            "--root_path",
            type=str,
            default="./dataset/ETT/",
            help="root path of the data file",
        )
        self.parser.add_argument(
            "--data_path", type=str, default="ETTh1.csv", help="data file"
        )
        self.parser.add_argument(
            "--features",
            type=str,
            default="M",
            help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, "
            "S:univariate predict univariate, MS:multivariate predict univariate",
        )
        self.parser.add_argument(
            "--target", type=str, default="OT", help="target feature in S or MS task"
        )
        self.parser.add_argument(
            "--freq",
            type=str,
            default="h",
            help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, "
            "b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
        )
        self.parser.add_argument(
            "--checkpoints",
            type=str,
            default="./checkpoints/",
            help="location of model checkpoints",
        )

        # forecasting task
        self.parser.add_argument(
            "--seq_len", type=int, default=96, help="input sequence length"
        )
        self.parser.add_argument(
            "--label_len", type=int, default=48, help="start token length"
        )
        self.parser.add_argument(
            "--pred_len", type=int, default=96, help="prediction sequence length"
        )

        # model define
        self.parser.add_argument(
            "--enc_in", type=int, default=7, help="encoder input size"
        )
        self.parser.add_argument(
            "--dec_in", type=int, default=7, help="decoder input size"
        )
        self.parser.add_argument("--c_out", type=int, default=7, help="output size")
        self.parser.add_argument(
            "--d_model", type=int, default=512, help="dimension of model"
        )
        self.parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
        self.parser.add_argument(
            "--e_layers", type=int, default=2, help="num of encoder layers"
        )
        self.parser.add_argument(
            "--d_layers", type=int, default=1, help="num of decoder layers"
        )
        self.parser.add_argument(
            "--d_ff", type=int, default=2048, help="dimension of fcn"
        )
        self.parser.add_argument(
            "--moving_avg", default=[24], help="window size of moving average"
        )
        self.parser.add_argument("--factor", type=int, default=1, help="attn factor")
        self.parser.add_argument(
            "--distil",
            action="store_false",
            help="whether to use distilling in encoder, using this argument means not using distilling",
            default=True,
        )
        self.parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
        self.parser.add_argument(
            "--embed",
            type=str,
            default="timeF",
            help="time features encoding, options:[timeF, fixed, learned]",
        )
        self.parser.add_argument(
            "--activation", type=str, default="gelu", help="activation"
        )
        self.parser.add_argument(
            "--output_attention",
            action="store_true",
            help="whether to output attention in ecoder",
        )
        self.parser.add_argument(
            "--do_predict",
            action="store_true",
            help="whether to predict unseen future data",
        )

        # optimization
        self.parser.add_argument(
            "--num_workers", type=int, default=10, help="data loader num workers"
        )
        self.parser.add_argument("--itr", type=int, default=3, help="experiments times")
        self.parser.add_argument(
            "--train_epochs", type=int, default=10, help="train epochs"
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=32, help="batch size of train input data"
        )
        self.parser.add_argument(
            "--patience", type=int, default=3, help="early stopping patience"
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.0001,
            help="optimizer learning rate",
        )
        self.parser.add_argument(
            "--des", type=str, default="test", help="exp description"
        )
        self.parser.add_argument(
            "--loss", type=str, default="mse", help="loss function"
        )
        self.parser.add_argument(
            "--lradj", type=str, default="type1", help="adjust learning rate"
        )
        self.parser.add_argument(
            "--use_amp",
            action="store_true",
            help="use automatic mixed precision training",
            default=False,
        )

        # GPU
        self.parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
        self.parser.add_argument("--gpu", type=int, default=0, help="gpu")
        self.parser.add_argument(
            "--use_multi_gpu",
            action="store_true",
            help="use multiple gpus",
            default=False,
        )
        self.parser.add_argument(
            "--devices", type=str, default="0,1", help="device ids of multi gpus"
        )

        # options coming from supervised transformer TODO check them in detail
        # Run from config file
        self.parser.add_argument(
            "--config",
            dest="config_filepath",
            help="Configuration .json file (optional). Overwrites existing command-line args!",
        )

        # Run from command-line arguments
        # I/O
        self.parser.add_argument(
            "--output_dir",
            default="./output",
            help="Root output directory. Must exist. Time-stamped directories will be created inside.",
        )
        self.parser.add_argument("--data_dir", default="./data", help="Data directory")
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument(
            "--name",
            dest="experiment_name",
            default="",
            help="A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp",
        )
        self.parser.add_argument(
            "--max_seq_len",
            type=int,
            help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""",
        )
        self.parser.add_argument(
            "--data_window_len",
            type=int,
            help="""Used instead of the `max_seq_len`, when the data samples must be
                                 segmented into windows. Determines maximum input sequence length 
                                 (size of transformer layers).""",
        )
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="only test model, no training",
        )
        self.parser.add_argument(
            "--print_interval",
            type=int,
            default=100,
            help="Print batch info every this many batches",
        )
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument(
            "--key_metric",
            choices={"loss", "accuracy", "precision"},
            default="loss",
            help="Metric used for defining best epoch",
        )
        self.parser.add_argument(
            "--save_all",
            action="store_true",
            help="If set, will save model weights (and optimizer state) for every epoch; otherwise just latest",
        )
        self.parser.add_argument(
            "--val_interval",
            type=int,
            default=2,
            help="Evaluate on validation set every this many epochs. Must be >= 1.",
        )

    def parse(self):

        args = self.parser.parse_args()

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(" ", "")
            device_ids = args.devices.split(",")
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        return args
