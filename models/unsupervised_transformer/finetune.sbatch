#!/bin/bash
#SBATCH --job-name="finetune unsupervised transformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/models/unsupervised_transformer/main.py --output_dir /usr/stud/roschman/ECGAnalysis/output/unsupervised_transformer --comment "finetune for classification" --name unsupervised_transformer_finetuned --records_file Classification_records.xls --epochs 100 --lr 0.001 --optimizer RAdam --batch_size 8 --pos_encoding learnable --d_model 256 --load_model /usr/stud/roschman/ECGAnalysis/output/unsupervised_transformer/unsupervised_transformer_pretrained_2022-10-20_00-41-13/checkpoints/model_best.pth --task classification --change_output --key_metric loss --max_seq_len 2500