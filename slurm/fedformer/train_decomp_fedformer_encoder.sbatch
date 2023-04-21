#!/bin/bash
#SBATCH --job-name="train decomp fedformer encoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name decomp_fedformer_encoder \
--num_layers 8 \
--num_heads 8 \
--d_model 256 \
--d_ff 512 \
--dropout 0.2 \
--activation gelu \
--version Fourier \
--modes 128 \
--mode_select random \
--moving_avg "[50, 100, 200]" \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.1 \
--epochs 100 \
--batch_size 16 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification