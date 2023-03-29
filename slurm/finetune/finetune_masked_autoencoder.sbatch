#!/bin/bash
#SBATCH --job-name="pretrain masked autoencoder"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name masked_autoencoder \
--mae \
--enc_num_layers 8 \
--enc_num_heads 8 \
--enc_d_model 256 \
--enc_d_ff 512 \
--dec_num_layers 5 \
--dec_num_heads 8 \
--dec_d_model 256 \
--dec_d_ff 512 \
--dropout 0.1 \
--head_dropout 0.1 \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--ch_token \
--cls_token \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 64 \
--num_workers 4 \
--patience 100 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification \
--finetuning \
--load_model /home/stud/roschman/ECGAnalysis/output/pretraining_patch_tst_2d/data=ecg_augment=False_randecg=_mixup=0_patch=16_stride=8_mratio=0.5_premratio=0_bs=32_opt=AdamW_lr=0.0001_wd=0.01_dmodel=256_dff=512_nlayers=8_nheads=8