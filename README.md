# Unsupervised Representation Learning for ECG Analysis

Machine learning algorithms can be trained to automatically detect and classify cardiac abnormalities given a patient's electrocardiogram (ECG).
Recent studies on multivariate time series have shown that the unsupervised pre-training of a model and its subsequent application in a classification task can outperform fully supervised approaches.
We investigate unsupervised pre-training for ECG analysis. In particular, we examine how human assumptions enhance or limit a model's capability to generalize and how well a model can classify cardiac abnormalities given differently large portions of ECG data.

## Dependencies
This repository works with Python 3.10 and PyTorch 1.13. Please create a virtual environment and install the dependencies specified in requirements.txt.

```bash
python3 -m venv ecg_analysis_env
source ecg_analysis_env/bin/activate
python3 -m pip install -r requirements.txt
```

## Data
We conduct most experiments on the ECG classification dataset provided for the PhysioNet Challenge 2020. However, our models can also be evaluated on forecasting
For each dataset, you can find a configuration file under data/configs.

### PhysioNet Challenge 2020
The dataset provided for the [PhysioNet Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) consists of 43101 12-lead ECG recordings from six different subsets. Data loading and preprocessing follow the provided baseline and the implementation of the winning approach.
All recordings are upsampled or downsampled to 100 Hz and can be optionally processed by a Finite Impulse Reponse bandpass filter with bandwith between 3 - 45 Hz to attenuate noise. Normalization ensures that the signal of each channel lies within the range of -1 to 1. We extract windows of fixed width T = 10s at a random position from each recording and apply zero padding for sequences of length less than T seconds. 

### Forecasting


### Augmentation

## Models
- **CNN Encoder**: We provide a simple CNN Encoder with 1D convolutions as a baseline.
- **CNN Transformer**: This model combines a CNN Encoder for feature extraction with a Transformer for feature enhancement. It is inspired by the [winning approach](https://ieeexplore.ieee.org/abstract/document/9344053) of the PhysioNet Challenge 2020. 
- **Transformer**: Zerveas et al. were the first to investigate unsupervised representation learning of multivariate time series with this Transformer encoder.
- **FEDformer**: This model applies attention operations in the frequency domain and achieves linear complexity by randomly selecting a fixed size subset of
frequencies. Proposed by Zhou et al. for time series forecasting, we adapt the model for time series classification.
- **PatchTST**: 
- **MaskedAutoencoder**: Inspired by the success of masked autoencoders for various computer vision tasks, we design a masked autoencoder for time series analysis. Our implementation is based on 

## Training
Please check out the directory /slurm with a variety of scripts to train the models on a SLURM-compatible compute system. In options.py you can see which parameters can be set for each training run. 
The models are specified with the options "--model_name". Supervised models can be trained for two tasks: 'classification' and 'forecasting'. Unsupervised models can be pretrained by specifying the task "pretraining" and finetuned by specifying one the two supervised tasks. Finetuning a model requires to load a pretrained model "--load_model" and set the flag "--finetuning".
To use a patched input, we set the flag "--use_patch" and specify the patch_len as well as the stride.


### Pretraining
The pretraining of our masked autoencoder for time series can be started as follows:

```bash
python3 main.py \
--model_name patch_tst \
--num_layers 8 \
--num_heads 8 \
--d_model 256 \
--d_ff 512 \
--dropout 0.1 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
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
--task pretraining
```

### Finetuning
To finetune the pretrained model on the ECG classification task, please use the following command:

```bash
python3 main.py \
--model_name patch_tst \
--num_layers 8 \
--num_heads 8 \
--d_model 256 \
--d_ff 512 \
--dropout 0.1 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
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
--task pretraining
```

## Results
