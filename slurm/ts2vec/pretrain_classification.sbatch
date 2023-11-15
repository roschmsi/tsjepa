#!/bin/bash
#SBATCH --job-name="pretrain ts2vec classification"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /home/stud/roschman/ECGAnalysis/ecg_env/bin/activate
srun python3 /home/stud/roschman/ECGAnalysis/pretrain_ts2vec.py \
--model_name ts2vec \
--enc_num_layers 8 \
--enc_num_heads 16 \
--enc_d_model 256 \
--enc_mlp_ratio 2 \
--dec_num_layers 2 \
--dec_num_heads 16 \
--dec_d_model 128 \
--dec_mlp_ratio 2 \
--predictor linear \
--mask_noise_std 0.02 \
--kernel_size 5 \
--dropout 0.2 \
--activation_drop_rate 0.2 \
--attn_drop_rate 0 \
--head_dropout 0 \
--norm LayerNorm \
--layer_norm_first \
--activation gelu \
--learn_pe \
--ema_decay 0.996 \
--ema_end_decay 1.0 \
--ema_anneal_end_step 2000 \
--average_top_k_layers 4 \
--normalize_targets \
--targets_norm LayerNorm \
--targets_rep encoder_out \
--masking random \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.00001 \
--epochs 2000 \
--weight_decay 0.01 \
--batch_size 512 \
--num_workers 4 \
--patience 2000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/ecg.yaml \
--channel_independence \
--task pretraining \
--output_dir /home/stud/roschman/ECGAnalysis/output \
--val_interval 10 \
--plot_interval 50 \
--pred_weight 1 \
--std_weight 0 \
--cov_weight 0 \
--loss smoothl1 \
--smoothl1_beta 0.5 \
--no_ema \
--description "rep=encout_mask=rand0.5_linear_smoothl1=0.5_std=1_cov=0.2_noema"