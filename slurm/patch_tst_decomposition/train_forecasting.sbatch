#!/bin/bash
#SBATCH --job-name="train patch tst decomposition"
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
--model_name patch_tst_decomposition \
--num_layers 4 \
--num_heads 8 \
--d_model 128 \
--d_ff 256 \
--dropout 0.3 \
--shared_embedding \
--norm LayerNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--learn_pe \
--patch_len 48 \
--stride 48 \
--optimizer AdamW \
--lr 0.00001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/weather.yaml \
--task forecasting \
--decomposition \
--separate_backbone \
--moving_avg 145 \
--seq_len 720 \
--label_len 0 \
--pred_len 96 \
--revin \
--val_interval 2 \
--description "decomp_module_mavg=145_mse"