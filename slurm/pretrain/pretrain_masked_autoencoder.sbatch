#!/bin/bash
#SBATCH --job-name="pretrain masked autoencoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--config_model /usr/stud/roschman/ECGAnalysis/models/masked_autoencoder/config_pretraining_masked_autoencoder.yaml \
--config_data /usr/stud/roschman/ECGAnalysis/data/ecg_dataset.yaml \
--load_model /home/stud/roschman/ECGAnalysis/output/pretraining_masked_autoencoder/2023-02-20_22-06-14_data=ecg_augment=False_mixup=0_patch=16_stride=8_mratio=0.75_bs=128_opt=AdamW_lr=0.0001_dropout=0.1 \
--resume