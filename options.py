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
        # self.parser.add_argument("--attn_dropout", type=float)
        self.parser.add_argument("--shared_embedding", action="store_true")
        self.parser.add_argument("--norm", type=str)
        self.parser.add_argument("--pre_norm", action="store_true")
        self.parser.add_argument("--activation", type=str)
        self.parser.add_argument("--learn_pe", action="store_true")

        # mae
        self.parser.add_argument("--mae", action="store_true")
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

        # trend-seasonal-residual decomposition
        self.parser.add_argument("--decomposition", action="store_true")
        self.parser.add_argument("--period", type=int)
        self.parser.add_argument("--trend_seasonal_residual", action="store_true")

        # tsjepa
        self.parser.add_argument("--ema_start", type=float)
        self.parser.add_argument("--ema_end", type=float)
        self.parser.add_argument("--no_ema", action="store_true")
        self.parser.add_argument("--load_encoder", type=str)
        self.parser.add_argument("--no_output_norm", action="store_true")
        self.parser.add_argument("--head_type", type=str)
        # self.parser.add_argument("--no_momentum", action="store_true")

        # ts2vec
        self.parser.add_argument("--activation_drop_rate", type=float)
        self.parser.add_argument("--layer_norm_first", action="store_true")
        self.parser.add_argument("--average_top_k_layers", type=int)
        self.parser.add_argument("--normalize_targets", action="store_true")
        self.parser.add_argument("--targets_norm", type=str)
        self.parser.add_argument("--normalize_pred", action="store_true")
        self.parser.add_argument("--pred_norm", type=str)
        self.parser.add_argument("--ema_decay", type=float)
        self.parser.add_argument("--ema_end_decay", type=float)
        self.parser.add_argument("--ema_anneal_end_step", type=int)
        self.parser.add_argument("--skip_embeddings", action="store_true")
        self.parser.add_argument("--skip_pos_embed", action="store_true")
        self.parser.add_argument("--skip_patch_embed", action="store_true")
        self.parser.add_argument("--targets_rep", type=str)
        self.parser.add_argument("--predictor", type=str)
        self.parser.add_argument("--kernel_size", type=int)
        self.parser.add_argument("--mask_noise_std", type=float)

        self.parser.add_argument("--step_size", type=int)

        self.parser.add_argument("--robustness", action="store_true")

        # stationarity
        self.parser.add_argument("--differencing", action="store_true")
        self.parser.add_argument("--lag", type=int, default=0)

        # vic regularization
        self.parser.add_argument("--vcreg", action="store_true")
        self.parser.add_argument("--vbcreg", action="store_true")
        self.parser.add_argument("--vic_reg_enc", action="store_true")
        self.parser.add_argument("--pred_weight", type=float)
        self.parser.add_argument("--std_weight", type=float)
        self.parser.add_argument("--cov_weight", type=float)

        self.parser.add_argument("--checkpoint_last", action="store_true")
        self.parser.add_argument("--checkpoint", type=int)

        # revserse instance normalization
        self.parser.add_argument("--revin", action="store_true")
        self.parser.add_argument("--patch_revin", action="store_true")
        self.parser.add_argument("--revin_affine", action="store_true")

        # decomposition
        self.parser.add_argument("--separate_backbone", action="store_true")

        # patch and mask
        self.parser.add_argument("--use_patch", action="store_true")
        self.parser.add_argument("--patch_len", type=int)
        self.parser.add_argument("--stride", type=int)
        self.parser.add_argument("--masking_ratio", type=float, default=0)
        self.parser.add_argument("--mean_mask_length", type=float)
        self.parser.add_argument("--cls_token", action="store_true")
        self.parser.add_argument("--ch_token", action="store_true")

        self.parser.add_argument("--masking", type=str)

        self.parser.add_argument("--bert", action="store_true")

        # hierarchical
        self.parser.add_argument("--hierarchical", action="store_true")
        self.parser.add_argument("--layer_wise_prediction", action="store_true")
        self.parser.add_argument("--num_levels", type=int)
        self.parser.add_argument("--window_size", type=int)
        # self.parser.add_argument(
        #     "--window_size",
        #     nargs="+",
        #     type=int,
        # )
        self.parser.add_argument("--ch_factor", type=float)
        self.parser.add_argument("--hierarchical_loss", action="store_true")
        self.parser.add_argument("--interpolation", action="store_true")
        self.parser.add_argument("--depths", nargs="+", type=int)
        self.parser.add_argument("--depths_encoder", nargs="+", type=int)
        self.parser.add_argument("--depths_decoder", nargs="+", type=int)
        self.parser.add_argument("--hierarchical_num_heads", nargs="+", type=int)

        # time covariates
        self.parser.add_argument("--use_time_features", action="store_true")
        self.parser.add_argument("--timeenc", type=int, default=0)
        self.parser.add_argument("--d_temp", type=int, default=0)
        self.parser.add_argument("--temporal_attention", action="store_true")
        self.parser.add_argument("--add_time_encoding", action="store_true")
        self.parser.add_argument("--concat_time_encoding", action="store_true")
        self.parser.add_argument("--patch_time", action="store_true")
        self.parser.add_argument("--input_time", action="store_true")

        # fedformer
        self.parser.add_argument("--version", type=str)
        self.parser.add_argument("--mode_select", type=str)
        self.parser.add_argument("--modes", type=int)
        self.parser.add_argument("--base", type=str)
        self.parser.add_argument("--moving_avg")

        # dataset
        self.parser.add_argument("--data_config", type=str)
        self.parser.add_argument("--filter_bandwidth", action="store_true")
        self.parser.add_argument("--augment", action="store_true")
        self.parser.add_argument("--mixup", type=float)
        self.parser.add_argument("--rand_ecg", type=str, default="")
        self.parser.add_argument("--channel_independence", action="store_true")

        # forecasting
        self.parser.add_argument("--seq_len", type=int)
        self.parser.add_argument("--label_len", type=int)
        self.parser.add_argument("--pred_len", type=int)

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

        # distributed training
        self.parser.add_argument("--distributed", action="store_true")

    def parse(self):
        args = self.parser.parse_args()
        return args
