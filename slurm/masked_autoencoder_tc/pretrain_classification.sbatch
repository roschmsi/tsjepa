#!/bin/bash
#SBATCH --job-name="pretrain masked autoencoder tc"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name masked_autoencoder_tc \
--mae \
--enc_num_layers 8 \
--enc_num_heads 8 \
--enc_d_model 256 \
--enc_d_ff 512 \
--dec_num_layers 2 \
--dec_num_heads 8 \
--dec_d_model 256 \
--dec_d_ff 512 \
--dropout 0.1 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0.1 \
--masking_ratio 0.9 \
--use_patch \
--patch_len 8 \
--stride 8 \
--cls_token \
--optimizer AdamW \
--lr 0.0001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 32 \
--num_workers 4 \
--patience 50 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task pretraining