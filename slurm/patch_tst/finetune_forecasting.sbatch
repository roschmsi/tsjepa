#!/bin/bash
#SBATCH --job-name="finetune patch tst"
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
--model_name patch_tst \
--num_layers 3 \
--num_heads 16 \
--d_model 128 \
--d_ff 256 \
--dropout 0.2 \
--shared_embedding \
--norm LayerNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--patch_len 8 \
--stride 8 \
--ch_token \
--optimizer AdamW \
--lr 0.00001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task forecasting \
--seq_len 512 \
--label_len 0 \
--pred_len 96 \
--finetuning \
--freeze \
--freeze_epochs 100 \
--load_model /home/stud/roschman/ECGAnalysis/output/patch_tst/pretraining/dataset=ettm1_sl=512_pl=0_patch=8_stride=8_mratio=0.5_bs=64_opt=AdamW_sch=CosineAnnealingLR_lr=1e-05_wd=0.01_dmodel=128_dff=256_nlayers=3_nheads=16_drop=0.2_norm=LayerNorm_act=gelu_chtoken