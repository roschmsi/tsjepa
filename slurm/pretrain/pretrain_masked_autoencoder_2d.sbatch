#!/bin/bash
#SBATCH --job-name="pretrain masked autoencoder 2d"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name masked_autoencoder_2d \
--enc_d_model 256 \
--enc_d_ff 512 \
--enc_num_heads 8 \
--enc_num_layers 8 \
--dec_d_model 256 \
--dec_d_ff 512 \
--dec_num_heads 8 \
--dec_num_layers 8 \
--dropout 0.1 \
--head_dropout 0.1 \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 64 \
--num_workers 4 \
--patience 100 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/etth1.yaml \
--task pretraining