#!/bin/bash
#SBATCH --job-name="finetune patch tst tc"
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
--model_name patch_tst_tc \
--num_layers 8 \
--num_heads 8 \
--d_model 256 \
--d_ff 512 \
--dropout 0.1 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0.1 \
--masking_ratio 0 \
--use_patch \
--patch_len 8 \
--stride 8 \
--cls_token \
--optimizer AdamW \
--lr 0.001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 32 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification \
--finetuning \
--freeze \
--freeze_epochs 100 \
--load_model /home/stud/roschman/ECGAnalysis/output/patch_tst_tc/pretraining/dataset=ecg_patch=8_stride=8_mratio=0.5_bs=32_opt=AdamW_sch=CosineAnnealingLR_lr=0.0001_wd=0.01_dmodel=256_dff=512_nlayers=8_nheads=8_drop=0.1_norm=BatchNorm_act=gelu_clstoken