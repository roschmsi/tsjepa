#!/bin/bash
#SBATCH --job-name="finetune unsupervised transformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py --config /usr/stud/roschman/ECGAnalysis/models/transformer/config_finetuning_transformer.yaml --load_model /usr/stud/roschman/ECGAnalysis/output/transformer_pretraining/transformer_pretraining_2022-11-14_11-37-06_baseline/checkpoints/model_best.pth --change_output --description no_freeze