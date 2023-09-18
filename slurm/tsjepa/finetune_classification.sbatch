#!/bin/bash
#SBATCH --job-name="finetune tsjepa classification"
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
--enc_num_layers 8 \
--enc_num_heads 16 \
--enc_d_model 256 \
--enc_mlp_ratio 2 \
--dropout 0.1 \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.001 \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 10 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification \
--freeze \
--load_model /home/stud/roschman/ECGAnalysis/output/tsjepa/pretraining/dataset=ecg_patch=16_stride=16_mratio=0.5_bs=128_opt=AdamW_lr=1e-05_wd=0.01_pred=50.0_std=50.0_cov=1.0_enc_l=8_h=16_d=256_dec_l=2_h=16_d=256_drop=0.1_emastart=0.996_still_hpt \
--description "freeze_pred=std=50"