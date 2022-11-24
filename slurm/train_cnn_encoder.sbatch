#!/bin/bash
#SBATCH --job-name="train cnn encoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:8G
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py --config /usr/stud/roschman/ECGAnalysis/models/cnn_transformer/config_cnn_encoder.yaml --description=all_data