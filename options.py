import argparse


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Run pipeline for all models")

        self.parser.add_argument(
            "--description",
            default="",
            help="Add description of your experiment.",
        )

        # model modes
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="If set, will load `starting_epoch` and state of optimizer, besides model weights.",
        )
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="only test model, no training",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="debug model with single sample",
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

        # model parameters
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="pretraining_patch_tst_2d",
        )
        self.parser.add_argument(
            "--d_model",
            type=int,
            default=256,
        )
        self.parser.add_argument(
            "--d_ff",
            type=int,
            default=512,
        )
        self.parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
        )
        self.parser.add_argument(
            "--num_layers",
            type=int,
            default=8,
        )
        self.parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
        )
        self.parser.add_argument(
            "--head_dropout",
            type=float,
            default=0.1,
        )

        self.parser.add_argument(
            "--enc_d_model",
            type=int,
            default=256,
        )
        self.parser.add_argument(
            "--enc_d_ff",
            type=int,
            default=512,
        )
        self.parser.add_argument(
            "--enc_num_heads",
            type=int,
            default=8,
        )
        self.parser.add_argument(
            "--enc_num_layers",
            type=int,
            default=8,
        )
        self.parser.add_argument(
            "--dec_d_model",
            type=int,
            default=256,
        )
        self.parser.add_argument(
            "--dec_d_ff",
            type=int,
            default=512,
        )
        self.parser.add_argument(
            "--dec_num_heads",
            type=int,
            default=8,
        )
        self.parser.add_argument(
            "--dec_num_layers",
            type=int,
            default=8,
        )

        # patch and mask parameters
        self.parser.add_argument(
            "--use_patch",
            action="store_true",
        )
        self.parser.add_argument(
            "--mae",
            action="store_true",
        )
        self.parser.add_argument(
            "--patch_len",
            type=int,
        )
        self.parser.add_argument(
            "--stride",
            type=int,
        )
        self.parser.add_argument(
            "--masking_ratio",
            type=float,
        )
        self.parser.add_argument(
            "--cls_token",
            action="store_true",
        )
        self.parser.add_argument(
            "--ch_token",
            action="store_true",
        )

        # data parameters
        self.parser.add_argument(
            "--data_config",
            help="provide path to dataset configuration",
        )
        self.parser.add_argument(
            "--filter_bandwidth",
            action="store_true",
        )
        self.parser.add_argument(
            "--augment",
            action="store_true",
        )
        self.parser.add_argument(
            "--mixup",
            type=float,
            default=0,
        )
        self.parser.add_argument(
            "--rand_ecg",
            type=str,
            default="",
        )

        # training parameters
        self.parser.add_argument(
            "--optimizer",
            type=str,
            default="AdamW",
        )
        self.parser.add_argument("--scheduler", type=str, default=None)
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.0001,
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=500,
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
        )
        self.parser.add_argument(
            "--patience",
            type=int,
            default=100,
        )

        # task
        self.parser.add_argument(
            "--task", type=str, help="choose from: pretraining, classification"
        )
        self.parser.add_argument(
            "--finetuning",
            action="store_true",
        )

        # evaluation
        self.parser.add_argument(
            "--beta",
            type=int,
            default=2,
        )
        self.parser.add_argument(
            "--weights_file",
            type=int,
            default=2,
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default="/usr/stud/roschman/ECGAnalysis/output",
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
