import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv/traffic_sl=720_pl=0_pat=24_str=24_mratio=0.5_bs=1024_sch=CA_lr=1e-05_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_revin_enc_l=4_h=16_d=128_d=128_drop=0.2_norm=LayerNorm_from_scratch_ema_only_mlp_noinit cov enc train.csv"
    )

    step = df["Step"].to_numpy()
    value = df["Value"].to_numpy()

    plt.figure(figsize=(4, 4))
    plt.plot(step, value)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv",
            f"encoder_covariance_train.png",
        )
    )
    plt.close()

    df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv/traffic_sl=720_pl=0_pat=24_str=24_mratio=0.5_bs=1024_sch=CA_lr=1e-05_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_revin_enc_l=4_h=16_d=128_d=128_drop=0.2_norm=LayerNorm_from_scratch_ema_only_mlp_noinit cov enc val.csv"
    )

    step = df["Step"].to_numpy()
    value = df["Value"].to_numpy()

    plt.figure(figsize=(4, 4))
    plt.plot(step, value)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv",
            f"encoder_covariance_val.png",
        )
    )
    plt.close()

    df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv/traffic_sl=720_pl=0_pat=24_str=24_mratio=0.5_bs=1024_sch=CA_lr=1e-05_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_revin_enc_l=4_h=16_d=128_d=128_drop=0.2_norm=LayerNorm_from_scratch_ema_only_mlp_noinit std enc train.csv"
    )

    step = df["Step"].to_numpy()
    value = df["Value"].to_numpy()

    plt.figure(figsize=(4, 4))
    plt.plot(step, value)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv",
            f"encoder_std_train.png",
        )
    )
    plt.close()

    df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv/traffic_sl=720_pl=0_pat=24_str=24_mratio=0.5_bs=1024_sch=CA_lr=1e-05_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_revin_enc_l=4_h=16_d=128_d=128_drop=0.2_norm=LayerNorm_from_scratch_ema_only_mlp_noinit std enc val.csv"
    )

    step = df["Step"].to_numpy()
    value = df["Value"].to_numpy()

    plt.figure(figsize=(4, 4))
    plt.plot(step, value)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv",
            f"encoder_std_val.png",
        )
    )
    plt.close()

    df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv/traffic_sl=720_pl=0_pat=24_str=24_mratio=0.5_bs=1024_sch=CA_lr=1e-05_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_revin_enc_l=4_h=16_d=128_d=128_drop=0.2_norm=LayerNorm_from_scratch_ema_only_mlp_noinit loss train.csv"
    )

    step = df["Step"].to_numpy()
    value = df["Value"].to_numpy()

    plt.figure(figsize=(4, 4))
    plt.plot(step, value)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv",
            f"loss_train.png",
        )
    )
    plt.close()

    df = pd.read_csv(
        "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv/traffic_sl=720_pl=0_pat=24_str=24_mratio=0.5_bs=1024_sch=CA_lr=1e-05_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_revin_enc_l=4_h=16_d=128_d=128_drop=0.2_norm=LayerNorm_from_scratch_ema_only_mlp_noinit loss val.csv"
    )

    step = df["Step"].to_numpy()
    value = df["Value"].to_numpy()

    plt.figure(figsize=(4, 4))
    plt.ylim([0.07, 0.12])
    plt.plot(step, value)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "/home/stud/roschman/ECGAnalysis/output/tensorboard_csv",
            f"loss_val.png",
        )
    )
    plt.close()
