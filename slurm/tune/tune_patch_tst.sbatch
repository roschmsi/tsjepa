#!/bin/bash
#SBATCH --job-name="tune patch tst"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/tune.py --config_model /usr/stud/roschman/ECGAnalysis/models/patch_tst/config_patch_tst.yaml --config_data /usr/stud/roschman/ECGAnalysis/data/ecg_dataset.yaml