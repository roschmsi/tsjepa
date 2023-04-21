#!/bin/bash
#SBATCH --job-name="pretrain unsupervised transformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
#SBATCH --no-requeue
source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/main.py \
--model_name transformer \
--num_layers 4 \
--num_heads 8 \
--d_model 256 \
--d_ff 512 \
--dropout 0.2 \
--cls_token \
--optimizer AdamW \
--masking_ratio 0 \
--lr 0.0001 \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task classification \
--finetuning \
--load_model /home/stud/roschman/ECGAnalysis/output/transformer/pretraining/dataset=ecg_mmlen=8.0_bs=32_opt=AdamW_lr=0.0001_wd=0.01_dmodel=256_dff=512_nlayers=4_nheads=8_drop=0.2_clstoken