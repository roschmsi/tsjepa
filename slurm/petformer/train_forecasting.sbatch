#!/bin/bash
#SBATCH --job-name="train petformer forecasting"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name petformer \
--num_layers 4 \
--num_heads 16 \
--d_model 438 \
--d_ff 1024 \
--dropout 0.5 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.0001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 128 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task forecasting \
--seq_len 720 \
--label_len 0 \
--pred_len 96 \
--learn_pe \
--val_interval 2 \
--revin \
--use_time_features \
--d_temp 74 \
--description "one_hot_time"