#!/bin/bash
#SBATCH --job-name="train hierarchical patch tst"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name hierarchical_patch_tst \
--num_levels 3 \
--enc_num_layers 1 \
--enc_num_heads 8 \
--enc_d_model 256 \
--enc_d_ff 512 \
--dec_num_layers 1 \
--dec_num_heads 8 \
--dec_d_model 256 \
--dec_d_ff 512 \
--ch_factor 1 \
--window_size 2 2 \
--dropout 0.5 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--patch_len 48 \
--stride 48 \
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
--seq_len 672 \
--label_len 0 \
--pred_len 720 \
--val_interval 2 \
--revin \
--description "full_att_"