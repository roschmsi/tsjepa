#!/bin/bash
#SBATCH --job-name="pretrain tsjepa"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/pretrain_tsjepa.py \
--model_name tsjepa \
--enc_num_layers 3 \
--enc_num_heads 8 \
--enc_d_model 16 \
--enc_mlp_ratio 2 \
--dec_num_layers 1 \
--dec_num_heads 8 \
--dec_d_model 16 \
--dec_mlp_ratio 2 \
--dropout 0.1 \
--ema_start 0.996 \
--ema_end 1.0 \
--masking_ratio 0.5 \
--use_patch \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.001 \
--weight_decay 0.01 \
--epochs 5000 \
--batch_size 64 \
--num_workers 4 \
--patience 5000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg5.yaml \
--task pretraining