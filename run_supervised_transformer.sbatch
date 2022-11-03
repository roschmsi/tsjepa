#!/bin/bash
#SBATCH --job-name="train ecg model"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:8G
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py --config /usr/stud/roschman/ECGAnalysis/models/supervised_transformer/config.yaml