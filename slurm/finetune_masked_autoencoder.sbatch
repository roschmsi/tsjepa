#!/bin/bash
#SBATCH --job-name="finetune masked autoencoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--config_model /usr/stud/roschman/ECGAnalysis/models/masked_autoencoder/config_finetuning_masked_autoencoder.yaml \
--config_data /usr/stud/roschman/ECGAnalysis/data/ecg_dataset.yaml \
--load_model /home/stud/roschman/ECGAnalysis/output/pretraining_masked_autoencoder/2023-02-10_11-02-56_data=ecg_augment=False_mixup=0_patch=16_stride=16_mratio=0.3_bs=64_opt=AdamW_lr=5e-05_enclayers=8_encdmodel=256_declayers=2_decdmodel=256 \