#!/bin/bash
#SBATCH --job-name="pretrain ts2vec"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=simon.roschmann@tum.de  
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/ts2vec_main.py \
--model_name ts2vec \
--optimizer AdamW \
--lr 1 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 64 \
--num_workers 4 \
--patience 500 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task pretraining \
--seq_len 512 \
--label_len 0 \
--pred_len 0 \
--description "try_to_prevent_overfitting"