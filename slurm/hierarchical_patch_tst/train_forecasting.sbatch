#!/bin/bash
#SBATCH --job-name="train hierarchical patch tst"
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
--model_name hierarchical_patch_tst \
--num_levels 3 \
--num_layers 3 \
--num_heads 16 \
--d_model 128 \
--d_ff 64 \
--ch_factor 0.5 \
--window_size 2 \
--dropout 0.2 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--patch_len 12 \
--stride 12 \
--optimizer AdamW \
--lr 0.00005 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task forecasting \
--seq_len 336 \
--label_len 0 \
--pred_len 96 \
--description chf=05_lbl_hierarchical_loss_revin_is_back \
--learn_pe \
--val_interval 2 \
--layer_wise_prediction \
--hierarchical_loss \
--revin