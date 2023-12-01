#!/bin/bash
#SBATCH --job-name="test robustness forecasting"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:8G
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /home/stud/roschman/ECGAnalysis/ecg_env/bin/activate
srun python3 /home/stud/roschman/ECGAnalysis/test_robustness.py \
--model_name ts2vec \
--enc_num_layers 4 \
--enc_num_heads 16 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dropout 0.2 \
--activation_drop_rate 0.2 \
--attn_drop_rate 0 \
--head_type linear \
--head_dropout 0 \
--norm LayerNorm \
--layer_norm_first \
--activation gelu \
--learn_pe \
--use_patch \
--patch_len 24 \
--stride 24 \
--weight_decay 0.01 \
--scheduler CosineAnnealingLR \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 10 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/weather.yaml \
--seq_len 720 \
--label_len 0 \
--pred_len 96 \
--channel_independence \
--task forecasting \
--robustness \
--revin \
--val_interval 2 \
--output_dir /home/stud/roschman/ECGAnalysis/output \
--load_model /home/stud/roschman/ECGAnalysis/output/ts2vec/forecasting/electricity_sl=720_pl=96_pat=24_str=24_bs=64_sch=CA_lr=0.0001_wd=0.01_fr=100_ema_revin_enc_l=4_h=16_d=128_drop=0.2_learnpe_norm=LayerNorm_only_ema_full_freeze_load_correct_model_flly \
--description "electricity robustness test"