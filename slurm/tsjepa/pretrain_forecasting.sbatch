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
--enc_num_layers 8 \
--enc_num_heads 16 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dec_num_layers 2 \
--dec_num_heads 16 \
--dec_d_model 128 \
--dec_mlp_ratio 2 \
--dropout 0.1 \
--head_dropout 0 \
--ema_start 0.996 \
--ema_end 1.0 \
--masking random \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.00001 \
--weight_decay 0.01 \
--epochs 2000 \
--batch_size 512 \
--num_workers 4 \
--patience 2000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/forecasting_all.yaml \
--seq_len 512 \
--label_len 0 \
--pred_len 96 \
--channel_independence \
--task pretraining \
--vic_reg \
--pred_weight 1.0 \
--std_weight 1.0 \
--cov_weight 1.0 \
--description with_revin