#!/bin/bash
#SBATCH --job-name="pretrain masked autoencoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name hierarchical_masked_autoencoder \
--mae \
--enc_num_levels 2 \
--enc_num_layers 1 \
--enc_num_heads 16 \
--enc_d_model 128 \
--enc_d_ff 256 \
--dec_num_levels 2 \
--dec_num_layers 1 \
--dec_num_heads 16 \
--dec_d_model 128 \
--dec_d_ff 256 \
--dropout 0.2 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0.1 \
--masking_ratio 0.5 \
--use_patch \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 64 \
--num_workers 4 \
--patience 50 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task pretraining \
--seq_len 512 \
--label_len 0 \
--pred_len 0