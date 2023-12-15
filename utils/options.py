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
        self.parser.add_argument("--mlp_ratio", type=int)
        self.parser.add_argument("--num_heads", type=int)
        self.parser.add_argument("--num_layers", type=int)
        self.parser.add_argument("--dropout", type=float)
        self.parser.add_argument("--attn_drop_rate", type=float, default=0)
        self.parser.add_argument("--drop_path_rate", type=float, default=0)
        self.parser.add_argument("--head_dropout", type=float)
        self.parser.add_argument("--shared_embedding", action="store_true")
        self.parser.add_argument("--norm", type=str)
        self.parser.add_argument("--activation", type=str)
        self.parser.add_argument("--learn_pe", action="store_true")

        # mae
        self.parser.add_argument("--enc_d_model", type=int)
        self.parser.add_argument("--enc_d_ff", type=int)
        self.parser.add_argument("--enc_num_heads", type=int)
        self.parser.add_argument("--enc_num_layers", type=int)
        self.parser.add_argument("--enc_mlp_ratio", type=int)
        self.parser.add_argument("--dec_d_model", type=int)
        self.parser.add_argument("--dec_d_ff", type=int)
        self.parser.add_argument("--dec_num_heads", type=int)
        self.parser.add_argument("--dec_num_layers", type=int)
        self.parser.add_argument("--dec_mlp_ratio", type=int)

        # loss
        self.parser.add_argument("--loss", type=str)
        self.parser.add_argument("--smoothl1_beta", type=float)

        # transformer
        self.parser.add_argument("--activation_drop_rate", type=float)
        self.parser.add_argument("--layer_norm_first", action="store_true")

        # contextualized targets
        self.parser.add_argument("--targets_rep", type=str)
        self.parser.add_argument("--average_top_k_layers", type=int)
        self.parser.add_argument("--normalize_targets", action="store_true")
        self.parser.add_argument("--targets_norm", type=str)
        self.parser.add_argument("--normalize_pred", action="store_true")
        self.parser.add_argument("--pred_norm", type=str)

        # ema update
        self.parser.add_argument("--ema_decay", type=float)
        self.parser.add_argument("--ema_end_decay", type=float)
        self.parser.add_argument("--ema_anneal_end_step", type=int)
        self.parser.add_argument("--no_ema", action="store_true")
        self.parser.add_argument("--skip_embeddings", action="store_true")
        self.parser.add_argument("--skip_pos_embed", action="store_true")
        self.parser.add_argument("--skip_patch_embed", action="store_true")

        # linear, MLP, or Transformer predictor
        self.parser.add_argument("--predictor", type=str)

        # vic regularization
        self.parser.add_argument("--vcreg", action="store_true")
        self.parser.add_argument("--pred_weight", type=float)
        self.parser.add_argument("--std_weight", type=float)
        self.parser.add_argument("--cov_weight", type=float)

        # reversible instance normalization
        self.parser.add_argument("--revin", action="store_true")
        self.parser.add_argument("--revin_affine", action="store_true")

        # patch and mask
        self.parser.add_argument("--use_patch", action="store_true")
        self.parser.add_argument("--patch_len", type=int)
        self.parser.add_argument("--stride", type=int)
        self.parser.add_argument("--masking", type=str)
        self.parser.add_argument("--masking_ratio", type=float, default=0)

        # load model
        self.parser.add_argument("--checkpoint_last", action="store_true")
        self.parser.add_argument("--checkpoint", type=int)

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
        self.parser.add_argument("--use_time_features", action="store_true")
        self.parser.add_argument("--timeenc", type=int, default=0)

        # training
        self.parser.add_argument("--optimizer", type=str)
        self.parser.add_argument("--scheduler", type=str)
        self.parser.add_argument("--lr", type=float)
        self.parser.add_argument("--start_lr", type=float)
        self.parser.add_argument("--ref_lr", type=float)
        self.parser.add_argument("--final_lr", type=float)
        self.parser.add_argument("--start_factor", type=float)
        self.parser.add_argument("--warmup", type=int)
        self.parser.add_argument("--weight_decay", type=float)
        self.parser.add_argument("--epochs", type=int)
        self.parser.add_argument("--batch_size", type=int)
        self.parser.add_argument("--num_workers", type=int)
        self.parser.add_argument("--patience", type=int)

        # model modes
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument("--resume", action="store_true")
        self.parser.add_argument("--test", action="store_true")
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_argument("--finetuning", action="store_true")

        self.parser.add_argument("--freeze", action="store_true")
        self.parser.add_argument("--freeze_epochs", type=int)

        # task
        self.parser.add_argument(
            "--task",
            type=str,
            help="pretraining, classification, forecasting",
        )
        self.parser.add_argument("--robustness", action="store_true")

        # print parameters
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument("--val_interval", type=int, default=5)
        self.parser.add_argument("--plot_interval", type=int, default=5)
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
            default="/usr/stud/roschman/ECGAnalysis/evaluation/weights.csv",
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default="/usr/stud/roschman/ECGAnalysis/output",
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
