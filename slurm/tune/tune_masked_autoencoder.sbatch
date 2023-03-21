#!/bin/bash
#SBATCH --job-name="tune patch tst"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/tune.py --config_model /usr/stud/roschman/ECGAnalysis/models/masked_autoencoder/config_pretraining_masked_autoencoder.yaml --config_data /usr/stud/roschman/ECGAnalysis/data/ecg_dataset.yaml