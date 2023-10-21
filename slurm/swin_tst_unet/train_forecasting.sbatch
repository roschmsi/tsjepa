#!/bin/bash
#SBATCH --job-name="train swin tst unet forecasting"
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
--model_name swin_tst_unet \
--depths_encoder 1 1 1 \
--depths_decoder 1 1 1 \
--hierarchical_num_heads 8 8 8 \
--d_model 128 \
--mlp_ratio 2 \
--dropout 0.5 \
--attn_drop_rate 0.0 \
--drop_path_rate 0.0 \
--window_size 4 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--learn_pe \
--patch_len 4 \
--stride 4 \
--optimizer AdamW \
--lr 0.00005 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 32 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task forecasting \
--seq_len 720 \
--label_len 0 \
--pred_len 720 \
--revin \
--val_interval 2 \
--description "no_input_dropout_pe"