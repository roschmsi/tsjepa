#!/bin/bash
#SBATCH --job-name="distributed pretrain tsjepa forecasting"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12347
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /usr/stud/roschman/ECGAnalysis/ecg_310/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/pretrain_tsjepa_distributed.py \
--model_name tsjepa \
--enc_num_layers 4 \
--enc_num_heads 16 \
--enc_d_model 64 \
--enc_mlp_ratio 2 \
--dec_num_layers 1 \
--dec_num_heads 16 \
--dec_d_model 64 \
--dec_mlp_ratio 2 \
--dropout 0.1 \
--head_dropout 0 \
--ema_start 0.996 \
--ema_end 1.0 \
--masking random \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--optimizer AdamW \
--lr 0.00001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 3000 \
--batch_size 256 \
--num_workers 4 \
--patience 3000 \
--data_config /home/stud/roschman/ECGAnalysis/data/configs/weather.yaml \
--seq_len 512 \
--label_len 0 \
--pred_len 0 \
--channel_independence \
--task pretraining \
--vic_reg \
--pred_weight 25.0 \
--std_weight 25.0 \
--cov_weight 1.0 \
--revin \
--distributed \
--description "distributed_n=2_g=1_12gb"