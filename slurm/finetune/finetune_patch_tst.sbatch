#!/bin/bash
#SBATCH --job-name="finetune patch tst"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:8G
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name patch_tst \
--d_model 256 \
--d_ff 512 \
--num_heads 8 \
--num_layers 8 \
--dropout 0.1 \
--head_dropout 0.1 \
--masking_ratio 0 \
--use_patch \
--patch_len 16 \
--stride 16 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 32 \
--num_workers 4 \
--patience 100 \
--ch_token \
--cls_token \
--task classification \
--finetuning \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--load_model /home/stud/roschman/ECGAnalysis/output/patch_tst_pretraining/dataset=ecg_augment=False_mixup=0_randecg=_patch=16_stride=16_mratio=0.5_preratio=None_bs=64_opt=AdamW_sched=None_lr=0.0001_wd=0.01_dmodel=256_dff=512_nlayers=8_nheads=8_chtoken=True \