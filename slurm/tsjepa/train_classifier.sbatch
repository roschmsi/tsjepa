#!/bin/bash
#SBATCH --job-name="train tsjepa classifier"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/train_tsjepa.py \
--model_name tsjepa \
--enc_num_layers 8 \
--enc_num_heads 8 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dropout 0.1 \
--head_dropout 0.0 \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.001 \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 128 \
--num_workers 4 \
--patience 50 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg_gender.yaml \
--task classification