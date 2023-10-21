#!/bin/bash
#SBATCH --job-name="finetune tsjepa forecasting"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/finetune_tsjepa.py \
--model_name tsjepa \
--enc_num_layers 4 \
--enc_num_heads 8 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dropout 0.2 \
--head linear \
--head_dropout 0 \
--norm LayerNorm \
--masking_ratio 0 \
--use_patch \
--patch_len 48 \
--stride 48 \
--optimizer AdamW \
--lr 0.00001 \
--learn_pe \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 128 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/weather.yaml \
--seq_len 720 \
--label_len 0 \
--pred_len 96 \
--channel_independence \
--task forecasting \
--val_interval 2 \
--revin \
--loss mse \
--load_model /home/stud/roschman/ECGAnalysis/output/tsjepa/pretraining/dataset=weather_sl=720_pl=0_patch=48_stride=48_mratio=0.5_bs=512_opt=AdamW_lr=1e-05_wd=0.1_pred=25.0_std=25.0_cov=10.0_revin_enc_l=4_h=8_d=128_dec_l=1_h=8_d=128_drop=0.2_norm=LayerNorm_act=gelu_emastart=0.996_nosched \
--description finally48pretrained