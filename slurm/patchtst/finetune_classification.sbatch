#!/bin/bash
#SBATCH --job-name="finetune patchtst classification"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /home/stud/roschman/ECGAnalysis/ecg_env/bin/activate
srun python3 /home/stud/roschman/ECGAnalysis/main_finetune.py \
--model_name patchtst \
--enc_num_layers 8 \
--enc_num_heads 16 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dropout 0.2 \
--activation_drop_rate 0.2 \
--attn_drop_rate 0 \
--head_dropout 0 \
--norm LayerNorm \
--layer_norm_first \
--activation gelu \
--learn_pe \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--scheduler CosineAnnealingLR \
--epochs 100 \
--batch_size 32 \
--num_workers 4 \
--patience 10 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification \
--val_interval 2 \
--output_dir /home/stud/roschman/ECGAnalysis/output