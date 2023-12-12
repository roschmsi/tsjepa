# Self-Supervised Representation Learning for Time Series Analysis

Time series data is prevalent across various domains including healthcare, transportation and manufacturing. The development of supervised machine learning models has lead to significant advancements in time series forecasting and classification. 
However, recent studies have shown that the self-supervised pre-training of a model can surpass fully supervised approaches in downstream tasks. 
A popular paradigm of self-supervised representation learning (SSL) is masked modeling where parts of the input signal are masked and the model is pre-trained to predict the masked parts given the unmasked context.

We propose a Time Series Joint Embedding Predictive Architecture (TS-JEPA), utilizing a self-distillation student-teacher setup.
While prior work has performed masked modeling in the input space, we are the first to investigate masked modeling for time series in the embedding space. A teacher encoder operates on the full input to generate contextualized target embeddings for the masked parts. Subsequently, a student encoder extracts meaningful representations from the unmasked input for the prediction of these target embeddings.

We investigate the parameterization of the teacher as a moving average of the student, apply different masking strategies during pre-training and explore the efficacy of variance-covariance regularization to prevent mode collapse.
We report the performance of our SSL approach in time series forecasting and classification on four public benchmarks. Additionally, we check the robustness of the extracted representations against input perturbation and the utility of TS-JEPA for transfer learning.

## Dependencies
This repository works with Python 3.10 and PyTorch 1.13. Please create a virtual environment and install the dependencies specified in requirements.txt.

```bash
python3 -m venv ecg_analysis_env
source ecg_analysis_env/bin/activate
python3 -m pip install -r requirements.txt
```

## Data
We conduct experiments on the ECG classification dataset provided for the PhysioNet Challenge 2020 and on eight popular forecasting datasets.
For each dataset, you can find a configuration file under `data/configs/`.

### ECG Classification
The [PhysioNet/Computing in Cardiology Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) provided 43101 12-lead ECG recordings from six different sources. The sampling frequency of the recordings varies from 257 Hz to 1000 Hz, their individual length lies between 6 seconds and 30 minutes. In total, there are 111 labeled abnormalities, 27 of which are included in the official scoring metric.
Hence, we associate each recording with a multilabel target comprising 27 clinical diagnoses. We extract windows of 10 seconds at a random position from each recording and apply zero padding for smaller sequences. We sample the time series signal at a frequency of 100 Hz and normalize the ECG recordings, such that the signal of each channel lies within the range of -1 to 1. Following [Natarajan et al.](https://www.cinc.org/2020/Program/accepted/107_CinCFinalPDF.pdf), we provide the option to apply a Finite Impulse Reponse (FIR) bandpass filter with a bandwith between 3 - 45 Hz to attenuate noise. The dataset is split into train, validation, and test sets in an 8:1:1 ratio.

### Time Series Forecasting
We select eight multivariate datasets which are publicly available at the [Autoformer repository](https://github.com/thuml/Autoformer) and have been extensively used to benchmark the performance of forecasting algorithms. 
* The [Weather](https://www.bgc-jena.mpg.de/wetter/) dataset comprises 21 meteorological indicators, including air temperature, humidity, and air pressure, recorded in a German city at a temporal resolution of every 10 minutes throughout the entire year of 2020.
* The [Traffic](https://pems.dot.ca.gov/) dataset is provided by the California Department of Transportation and contains hourly road occupancy rates measured by various sensors deployed on freeways in the San Francisco Bay area.
* The [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) dataset includes hourly electricity consumption data from 321 customers over three entire years, spanning from 2012 to 2014.
* The [ILI](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html) dataset is recorded by the Centers for Disease Control and Prevention of the United States. It contains records of the total number of patients and the ratio of patients with influenza-like illness (ILI) in different age groups on a weekly basis between 2002 to 2021.
* The [ETT](https://github.com/zhouhaoyi/ETDataset) datasets depict the load and oil temperature of electric transformers. They are collected from two different devices, denoted as transformer 1 and 2, at two different resolutions (15 minutes and 1 hour), denoted as m and h, between July 2016 and July 2018. In total, this recording scheme yields four ETT datasets, namely ETTm1, ETTm2, ETTh1, and ETTh2.

Following [Wu et al.](https://arxiv.org/abs/2106.13008), the datasets are split into train, validation, and test sets in chronological order, with a ratio of 6:2:2 for the ETT dataset and 7:1:2 for the other datasets.


## Models
Inspired by the success of masked modeling in NLP and CV, [Nie et al.](https://arxiv.org/pdf/2211.14730.pdf) proposed the **Patch Time Series Transformer (PatchTST)**. Pre-training this Transformer model on time series patches according to the masked modeling paradigm yields state-of-the-art results in time series forecasting with a notable improvement over the fully supervised approach.

To efficiently pre-train large Transformer models on images, [He et al.](https://arxiv.org/abs/2111.06377) recently introduced **Masked Autoencoders (MAE)**. This methods involves dividing an image into patches, with a substantial portion of patches being masked during pre-training. The encoder only operates on the visible subset of patches while the decoder reconstructs the original image from the latent representation and mask tokens. 


![](img/tsjepa.png)
Pre-training setup of **PatchTST** and **MAE**. PatchTST (left, adapted from [Nie et al.](https://arxiv.org/abs/2211.14730)) initially segments the time series signal into patches. A significant amount of these patches is masked (grey). All patches are linearly projected into an embedding space and provided with a positional encoding. The sequence is processed by a Transformer block. A final linear projection predicts the signal of the masked patches. 
MAE (right, inspired by [He et al.](https://arxiv.org/abs/2111.06377)) is an encoder-decoder architecture. In contrast to PatchTST, the masked patches are discarded and only the unmasked patches (green) are forwarded through the MAE encoder. We obtain a latent representation of the visible patches. This latent representation is unshuffled and refilled with learnable mask tokens (orange) which indicate the presence of masked patches. The entire sequence is fed into the MAE decoder, which is trained to predict the signal of the masked patches.


## Training
The `slurm/` folder contains scripts to train the models on a SLURM-compatible compute system. Check out `options.py` to gain an overview of the training parameters.

To train a particular model, you have to provide the `--model_name` and define its configuration. Supervised models can be trained for two tasks: `classification` or `forecasting`. If you are interested in unsupervised models, you can pre-train them by selecting the task `pretraining`. Once you have a pre-trained model, you can fine-tune it for a supervised task. You need to load the pre-trained model with `--load_model` and set the `--finetuning` flag.

To patch the time series input, please set the `--use_patch` flag and specify `--patch_len` and `--stride`.


### Pre-training
The pre-training of our MAE for time series with a masking ratio of 50 % and non-overlapping patches of size 8 can be initiated as follows:

```bash
python3 main.py \
--model_name masked_autoencoder \
--mae \
--enc_num_layers 8 \
--enc_num_heads 8 \
--enc_d_model 256 \
--enc_d_ff 512 \
--dec_num_layers 2 \
--dec_num_heads 8 \
--dec_d_model 256 \
--dec_d_ff 512 \
--dropout 0.2 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0.1 \
--masking_ratio 0.5 \
--use_patch \
--patch_len 8 \
--stride 8 \
--cls_token \
--optimizer AdamW \
--lr 0.0001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 500 \
--batch_size 64 \
--num_workers 4 \
--patience 50 \
--data_config data/configs/ecg.yaml \
--task pretraining
```

### Finetuning
To finetune the pre-trained MAE for ECG classification, please adapt the following command:

```bash
python3 main.py \
--model_name masked_autoencoder \
--mae \
--enc_num_layers 8 \
--enc_num_heads 8 \
--enc_d_model 256 \
--enc_d_ff 512 \
--dec_num_layers 2 \
--dec_num_heads 8 \
--dec_d_model 256 \
--dec_d_ff 512 \
--dropout 0.1 \
--shared_embedding \
--norm BatchNorm \
--activation gelu \
--head_dropout 0.1 \
--masking_ratio 0 \
--use_patch \
--patch_len 8 \
--stride 8 \
--ch_token \
--cls_token \
--optimizer AdamW \
--lr 0.0001 \
--scheduler CosineAnnealingLR \
--weight_decay 0.01 \
--epochs 100 \
--batch_size 64 \
--num_workers 4 \
--patience 20 \
--data_config data/configs/ecg.yaml \
--task classification \
--finetuning \
--freeze \
--freeze_epochs 100 \
--load_model "path to pre-trained model"
```


## Acknowledgements

We appreciate the following websites and repositories for their valuable code base and datasets:

* https://physionet.org/content/challenge-2020/1.0.2/
* https://github.com/physionetchallenges/evaluation-2020
* https://github.com/yuqinie98/PatchTST
* https://github.com/facebookresearch/mae
* https://github.com/gzerveas/mvts_transformer
* https://github.com/MAZiqing/FEDformer
* https://github.com/thuml/Autoformer
* https://github.com/seitalab/RandECG


## Contact

If you have any questions or concerns, please feel free to contact me: simon.roschmann@tum.de