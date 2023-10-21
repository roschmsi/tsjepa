#!/bin/bash
#SBATCH --job-name="finetune tsjepa classification"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/finetune_tsjepa.py \
--model_name tsjepa \
--enc_num_layers 8 \
--enc_num_heads 8 \
--enc_d_model 256 \
--enc_mlp_ratio 2 \
--dropout 0.1 \
--head_dropout 0 \
--masking_ratio 0 \
--norm BatchNorm \
--use_patch \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 128 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification \
--val_interval 2 \
--description no_prtr