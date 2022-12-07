#!/bin/bash
#SBATCH --job-name="finetune unsupervised transformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py --config_model /usr/stud/roschman/ECGAnalysis/models/transformer/config_finetuning_transformer.yaml --config_data /usr/stud/roschman/ECGAnalysis/data/uea_dataset.yaml --load_model /usr/stud/roschman/ECGAnalysis/output/pretraining_transformer/pretraining_transformer_2022-12-01_08-42-28/checkpoints/model_best.pth --change_output --description no_freeze