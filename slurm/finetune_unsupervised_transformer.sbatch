#!/bin/bash
#SBATCH --job-name="finetune unsupervised transformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py --config /usr/stud/roschman/ECGAnalysis/models/unsupervised_transformer/config_finetuning.yaml --load_model /usr/stud/roschman/ECGAnalysis/output/unsupervised_transformer/unsupervised_transformer_2022-11-03_11-55-40/checkpoints/model_best.pth --change_output