#!/bin/bash
#SBATCH --job-name="pretrain tsjepa forecasting"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/pretrain_tsjepa.py \
--model_name tsjepa \
--enc_num_layers 4 \
--enc_num_heads 8 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dec_num_layers 1 \
--dec_num_heads 8 \
--dec_d_model 128 \
--dec_mlp_ratio 2 \
--dropout 0.2 \
--head_dropout 0 \
--norm LayerNorm \
--activation gelu \
--ema_start 0.996 \
--ema_end 1.0 \
--masking random \
--masking_ratio 0.5 \
--use_patch \
--patch_len 24 \
--stride 24 \
--optimizer AdamW \
--lr 0.0001 \
--scheduler CosineAnnealingLRWithWarmup \
--warmup 200 \
--start_factor 0.1 \
--weight_decay 0.1 \
--epochs 1000 \
--batch_size 512 \
--num_workers 4 \
--patience 2000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/weather.yaml \
--seq_len 720 \
--label_len 0 \
--pred_len 0 \
--channel_independence \
--task pretraining \
--vic_reg \
--pred_weight 25 \
--std_weight 25 \
--cov_weight 5 \
--revin \
--val_interval 10 \
--plot_interval 50 \
--no_ema \
--no_output_norm \
--description "no_ema_no_output_norm"