#!/bin/bash
#SBATCH --job-name="finetune ts2vec forecasting"
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
--enc_num_layers 4 \
--enc_num_heads 8 \
--enc_d_model 128 \
--enc_mlp_ratio 2 \
--dropout 0.2 \
--attn_drop_rate 0 \
--activation_drop_rate 0 \
--head_type linear \
--head_dropout 0 \
--norm LayerNorm \
--layer_norm_first \
--activation gelu \
--learn_pe \
--use_patch \
--patch_len 48 \
--stride 48 \
--optimizer AdamW \
--lr 0.00005 \
--weight_decay 0 \
--scheduler CosineAnnealingLR \
--epochs 100 \
--batch_size 128 \
--num_workers 4 \
--patience 20 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/weather.yaml \
--seq_len 720 \
--label_len 0 \
--pred_len 720 \
--channel_independence \
--task forecasting \
--revin \
--val_interval 2 \
--output_dir /home/stud/roschman/ECGAnalysis/output \
--load_model /home/stud/roschman/ECGAnalysis/output/ts2vec/pretraining/dataset=weather_sl=720_pl=0_patch=24_stride=24_mratio=0.5_bs=1024_opt=AdamW_sch=CA_lr=1e-05_wd=0.04_revin_enc_l=4_h=8_d=128_drop=0.2_learnpe_norm=LayerNorm_act=gelu_skip_embeddings \
--checkpoint_last \
--freeze \
--freeze_epochs 20 \
--description "layers=4_skip_embeddings"