#!/bin/bash
#SBATCH --job-name="finetune patch tst 2d"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name patch_tst_2d \
--d_model 256 \
--d_ff 512 \
--num_heads 8 \
--num_layers 8 \
--dropout 0.1 \
--head_dropout 0.1 \
--masking_ratio 0 \
--masking_ratio_pretraining 0.5 \
--use_patch \
--patch_len 16 \
--stride 8 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 32 \
--num_workers 4 \
--patience 100 \
--task classification \
--finetuning \
--dataset ecg \
--window 10 \
--fs 100 \
--num_classes 27 \
--feat_dim 12 \
--multilabel \
--augment \
--load_model /home/stud/roschman/ECGAnalysis/output/pretraining_patch_tst_2d/data=ecg_augment=False_randecg=_mixup=0_patch=16_stride=8_mratio=0.5_premratio=0_bs=32_opt=AdamW_lr=0.0001_wd=0.01_dmodel=256_dff=512_nlayers=8_nheads=8 \