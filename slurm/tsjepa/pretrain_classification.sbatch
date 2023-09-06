#!/bin/bash
#SBATCH --job-name="pretrain tsjepa classification"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
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
--enc_d_model 256 \
--enc_mlp_ratio 2 \
--dec_num_layers 2 \
--dec_num_heads 16 \
--dec_d_model 256 \
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
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 1000 \
--batch_size 128 \
--num_workers 4 \
--patience 1000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task pretraining \
--vic_reg \
--pred_weight 25.0 \
--std_weight 25.0 \
--cov_weight 1.0 \
--description "still_hpt"