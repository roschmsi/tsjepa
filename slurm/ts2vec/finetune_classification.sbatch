#!/bin/bash
#SBATCH --job-name="finetune ts2vec classification"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /home/stud/roschman/ECGAnalysis/ecg_env/bin/activate
srun python3 /home/stud/roschman/ECGAnalysis/finetune_ts2vec.py \
--model_name ts2vec \
--enc_num_layers 8 \
--enc_num_heads 16 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dropout 0.2 \
--activation_drop_rate 0.2 \
--attn_drop_rate 0 \
--head_type linear \
--head_dropout 0 \
--norm LayerNorm \
--layer_norm_first \
--activation gelu \
--learn_pe \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.0001 \
--weight_decay 0.01 \
--scheduler CosineAnnealingLR \
--epochs 100 \
--batch_size 32 \
--num_workers 4 \
--patience 10 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--channel_independence \
--task classification \
--val_interval 2 \
--output_dir /home/stud/roschman/ECGAnalysis/output \
--load_model /home/stud/roschman/ECGAnalysis/output/ts2vec/pretraining/ecg_pat=8_str=8_mratio=0.5_bs=1024_sch=CA_lr=0.0001_wd=0.01_pred=1.0_std=0.0_cov=0.0_ema_enc_l=8_h=16_d=128_dec_l=2_h=16_d=128_drop=0.2_norm=LayerNorm_mae \
--description "mae_finetune_result_physionet"