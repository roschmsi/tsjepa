#!/bin/bash
#SBATCH --job-name="finetune masked autoencoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py --config_model /usr/stud/roschman/ECGAnalysis/models/masked_autoencoder/config_finetuning_masked_autoencoder.yaml --config_data /usr/stud/roschman/ECGAnalysis/data/ecg_dataset.yaml --load_model /usr/stud/roschman/ECGAnalysis/output/pretraining_patch_tst/2023-01-03_09-27-02_data=ptb-xl_window=10_fs=100_augment=True_mixup=0_bs=128_opt=Adam_lr=0.0001_dmodel=256_dff=512_nlayers=8_nheads=8_dropout=0.1_maskratio=0.4/checkpoints/model_best.pth --change_output