import argparse


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Run pipeline for all models")

        self.parser.add_argument(
            "--description",
            type=str,
            default="",
            help="Add description of your experiment.",
        )

        # transformer
        self.parser.add_argument("--model_name", type=str)
        self.parser.add_argument("--num_cnn", type=int)
        self.parser.add_argument("--d_model", type=int)
        self.parser.add_argument("--d_ff", type=int)
        self.parser.add_argument("--num_heads", type=int)
        self.parser.add_argument("--num_layers", type=int)
        self.parser.add_argument("--dropout", type=float)
        self.parser.add_argument("--head_dropout", type=float)
        self.parser.add_argument("--shared_embedding", action="store_true")
        self.parser.add_argument("--norm", type=str)
        self.parser.add_argument("--activation", type=str)
        self.parser.add_argument("--learn_pe", action="store_true")

        # mae
        self.parser.add_argument("--mae", action="store_true")
        self.parser.add_argument("--enc_d_model", type=int)
        self.parser.add_argument("--enc_d_ff", type=int)
        self.parser.add_argument("--enc_num_heads", type=int)
        self.parser.add_argument("--enc_num_layers", type=int)
        self.parser.add_argument("--dec_d_model", type=int)
        self.parser.add_argument("--dec_d_ff", type=int)
        self.parser.add_argument("--dec_num_heads", type=int)
        self.parser.add_argument("--dec_num_layers", type=int)

        # patch and mask
        self.parser.add_argument("--use_patch", action="store_true")
        self.parser.add_argument("--patch_len", type=int)
        self.parser.add_argument("--stride", type=int)
        self.parser.add_argument("--masking_ratio", type=float, default=0)
        self.parser.add_argument("--mean_mask_length", type=float)
        self.parser.add_argument("--cls_token", action="store_true")
        self.parser.add_argument("--ch_token", action="store_true")

        # fedformer
        self.parser.add_argument("--version", type=str)
        self.parser.add_argument("--mode_select", type=str)
        self.parser.add_argument("--modes", type=int)
        self.parser.add_argument("--moving_avg")

        # dataset
        self.parser.add_argument("--data_config", type=str)
        self.parser.add_argument("--filter_bandwidth", action="store_true")
        self.parser.add_argument("--augment", action="store_true")
        self.parser.add_argument("--mixup", type=float)
        self.parser.add_argument("--rand_ecg", type=str, default="")

        # forecasting
        self.parser.add_argument("--seq_len", type=int)
        self.parser.add_argument("--label_len", type=int)
        self.parser.add_argument("--pred_len", type=int)

        # training
        self.parser.add_argument("--optimizer", type=str)
        self.parser.add_argument("--scheduler", type=str)
        self.parser.add_argument("--lr", type=float)
        self.parser.add_argument("--weight_decay", type=float)
        self.parser.add_argument("--epochs", type=int)
        self.parser.add_argument("--batch_size", type=int)
        self.parser.add_argument("--num_workers", type=int)
        self.parser.add_argument("--patience", type=int)

        # model modes
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument("--resume", action="store_true")
        self.parser.add_argument("--freeze", action="store_true")
        self.parser.add_argument("--test", action="store_true")
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_argument("--finetuning", action="store_true")

        # task
        self.parser.add_argument(
            "--task",
            type=str,
            help="pretraining, classification, forecasting",
        )

        # print parameters
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument(
            "--val_interval",
            type=int,
            default=2,
            help="Evaluate on validation set every this many epochs. Must be >= 1.",
        )
        self.parser.add_argument(
            "--print_interval",
            type=int,
            default=100,
            help="Print batch info every this many batches",
        )

        # evaluation
        self.parser.add_argument(
            "--beta",
            type=int,
            default=2,
        )
        self.parser.add_argument(
            "--weights_file",
            type=str,
            default="/usr/stud/roschman/ECGAnalysis/physionet_evaluation/weights.csv",
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default="/usr/stud/roschman/ECGAnalysis/output",
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
