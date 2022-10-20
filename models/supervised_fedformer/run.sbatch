#!/bin/bash
#SBATCH --job-name="train supervised fedformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
source /usr/stud/roschman/ECGAnalysis/ecg_analysis_env/bin/activate
srun python3 /usr/stud/roschman/ECGAnalysis/models/supervised_fedformer/run.py --is_training 1 --root_path None --data_path None --task_id ECG --model FEDformerECG --data ECG --features None --seq_len 2500 --label_len 2500 --pred_len 0 --e_layers 2 --d_layers 1 --factor 3 --enc_in 12 --dec_in 12 --c_out 12 --des 'Exp' --d_model 256 --itr 1 --batch_size 16 --train_epochs 100 --checkpoints /usr/stud/roschman/ECGAnalysis/output/supervised_fedformer