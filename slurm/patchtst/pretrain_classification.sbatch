#!/bin/bash
#SBATCH --job-name="pretrain patchtst classification"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=16G
#SBATCH --time=144:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /home/stud/roschman/ECGAnalysis/ecg_env/bin/activate
srun python3 /home/stud/roschman/ECGAnalysis/main_pretrain.py \
--model_name patchtst \
--enc_num_layers 8 \
--enc_num_heads 16 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--predictor linear \
--dropout 0.2 \
--activation_drop_rate 0.2 \
--attn_drop_rate 0 \
--head_dropout 0 \
--norm LayerNorm \
--layer_norm_first \
--activation gelu \
--masking random \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.0001 \
--scheduler CosineAnnealingLR \
--epochs 2000 \
--weight_decay 0.01 \
--batch_size 1024 \
--num_workers 4 \
--patience 2000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--task pretraining \
--output_dir /home/stud/roschman/ECGAnalysis/output \
--val_interval 10 \
--plot_interval 50 \
--pred_weight 1 \
--std_weight 0 \
--cov_weight 0 \
--loss mse