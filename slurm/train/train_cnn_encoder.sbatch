#!/bin/bash
#SBATCH --job-name="train cnn encoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name cnn_encoder \
--num_cnn 5 \
--d_model 256 \
--dropout 0.1 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 64 \
--num_workers 4 \
--patience 100 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification