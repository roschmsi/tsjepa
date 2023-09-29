#!/bin/bash
#SBATCH --job-name="train patch tst"
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
--d_model 64 \
--d_ff 128 \
--dropout 0.1 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0 \
--masking_ratio 0 \
--use_patch \
--learn_pe \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.00005 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 100 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ettm1.yaml \
--task forecasting \
--seq_len 513 \
--label_len 0 \
--pred_len 96 \
--differencing \
--description "differencing_test_2nd_try" \
--load_model /home/stud/roschman/ECGAnalysis/output/patch_tst/forecasting/dataset=ettm1_sl=513_pl=96_patch=8_stride=8_bs=64_opt=AdamW_sch=CosineAnnealingLR_lr=5e-05_wd=0.01_nlayers=3_nheads=16_dmodel=64_dff=128_drop=0.1_learnpe_norm=BatchNorm_act=gelu_differencing \
--test