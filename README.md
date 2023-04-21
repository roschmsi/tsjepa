# Unsupervised Representation Learning for ECG Analysis

Machine learning algorithms can be trained to automatically detect and classify cardiac abnormalities given a patient's electrocardiogram (ECG).Recent studies on multivariate time series have shown that the unsupervised pre-training of a model and its subsequent application in downstream tasks can outperform fully supervised approaches.

In this work, we investigate the effectivenss of unsupervised pre-training for ECG analysis. We identify the pivotal elements of masked time series modeling with Transformers and achieve state-of-the-art classification results on a large-scale ECG dataset. Additionally, we propose masked autoencoders for time series and demonstrate their applicability in classification and forecasting tasks.

## Dependencies
This repository works with Python 3.10 and PyTorch 1.13. Please create a virtual environment and install the dependencies specified in requirements.txt.

```bash
python3 -m venv ecg_analysis_env
source ecg_analysis_env/bin/activate
python3 -m pip install -r requirements.txt
```

## Data
We conduct most experiments on the ECG classification dataset provided for the PhysioNet Challenge 2020. However, our models can also be evaluated on eight popular forecasting datasets.
For each dataset, you can find a configuration file under data/configs.

### ECG Classification
The [PhysioNet/Computing in Cardiology Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) provided 43101 12-lead ECG recordings from six different sources. The sampling frequency of the recordings varies from 257 Hz to 1000 Hz, their individual length lies between 6 seconds and 30 minutes. In total, there are 111 labeled abnormalities, 27 of which are included in the official scoring metric.
Hence, we associate each recording with a multilabel target comprising 27 clinical diagnoses. We extract windows of $10 s$ at a random position from each recording and apply zero padding for smaller sequences. We sample the time series signal at a frequency of 100 Hz and normalize the ECG recordings, such that the signal of each channel lies within the range of -1 to 1. Following Natarajan et al. \cite{natarajan2020wideanddeep}, we provide the option to apply a Finite Impulse Reponse (FIR) bandpass filter with a bandwith between 3 - 45 Hz to attenuate noise. The dataset is split into train, validation, and test sets in an 8:1:1 ratio.

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
We are the first to conduct an in-depth investigation of the PatchTST components for time series classification. While Nie et al. assume channel independence in their time series forecasting experiments, we attempt to model inter-channel dependencies. Therefore, we introduce channel tokens to the PatchTST architecture and examine the utility of different patching strategies (patch_tst, patch_tst_t, patch_tst_tc) across time and channel dimension.

To efficiently pre-train large Transformer models on images, [He et al.](https://arxiv.org/abs/2111.06377) recently introduced **Masked Autoencoders (MAE)**. This methods involves dividing an image into patches, with a substantial portion of patches being masked during pre-training. The encoder only operates on the visible subset of patches while the decoder reconstructs the original image from the latent representation and mask tokens. 
To the best of our knowledge, we are the first to propose MAE for pre-training on time series patches. We investigate the modeling of inter-channel dependencies with our MAE (masked_autoencoder, masked_autoencoder_t, masked_autoencoder_tc) and evaluate its performance in large-scale ECG classification and time series forecasting.

![](img/masked_autoencoder.png)
Pre-training setup of **PatchTST** and **MAE**. PatchTST (left, adapted from [Nie et al.](https://arxiv.org/abs/2211.14730)) initially segments the time series signal into patches. A significant amount of these patches is masked (grey). All patches are linearly projected into an embedding space and provided with a positional encoding. The sequence is processed by a Transformer block. A final linear projection predicts the signal of the masked patches. 
MAE (right, inspired by [He et al.](https://arxiv.org/abs/2111.06377)) is an encoder-decoder architecture. In contrast to PatchTST, the masked patches are discarded and only the unmasked patches (green) are forwarded through the MAE encoder. We obtain a latent representation of the visible patches. This latent representation is unshuffled and refilled with learnable mask tokens (orange) which indicate the presence of masked patches. The entire sequence is fed into the MAE decoder, which is trained to predict the signal of the masked patches.

The performance of these Patch Transformers can be compared to a strong baseline of supervised classification approaches.
- **CNN Classifier**: We provide a simple CNN classifier comprising two convolutional layers with kernel size 11 and stride 3, two convolutional layers with kernel size 7 and stride 2 and one convolutional layer with kernel size 3 and stride 1.
- **CNN Transformer**: Inspired by the [winning approach](https://ieeexplore.ieee.org/abstract/document/9344053) of the PhysioNet Challenge 2020, this model also utilizes a CNN Encoder with the above-mentiond configuration and applies an additional Transformer on the extracted features for feature enhancement.
- **Transformer**: [Zerveas et al.](https://github.com/gzerveas/mvts_transformer) were the first to investigate unsupervised representation learning of multivariate time series with a Transformer encoder. While PatchTST and MAE apply patching, this approach directly masks parts of the raw time series input during pre-training. This Transformer model can also be trained on the the raw time series input in a supervised manner.
- **FEDformer**: This model applies attention operations in the frequency domain and achieves linear complexity by randomly selecting a fixed size subset of
frequencies. Proposed by Zhou et al. for time series forecasting, we adapt the model for time series classification.

## Training
Please check out the directory ==/slurm== with a variety of scripts to train the models on a SLURM-compatible compute system. In options.py you can see which parameters can be set for each training run. 
The models are specified with the options "--model_name". Supervised models can be trained for two tasks: 'classification' and 'forecasting'. Unsupervised models can be pretrained by specifying the task "pretraining" and finetuned by specifying one the two supervised tasks. Finetuning a model requires to load a pretrained model "--load_model" and set the flag "--finetuning".
To use a patched input, we set the flag "--use_patch" and specify the patch_len as well as the stride.


### Pretraining
The pretraining of our MAE for time series with a masking ratio of 50 % and non-overlapping patches of size 8 can be started as follows:

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
To finetune the pre-trained MAE for ECG classification, please use the following command:

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
--load_model "path to pretrained model"
```


## Acknowledgement

We appreciate the following websites and repositories for their valuable code base and datasets:

https://physionet.org/content/challenge-2020/1.0.2/

https://github.com/yuqinie98/PatchTST

https://github.com/facebookresearch/mae

https://github.com/gzerveas/mvts_transformer

https://github.com/MAZiqing/FEDformer

https://github.com/thuml/Autoformer
https://github.com/seitalab/RandECG


## Contact

If you have any questions or concerns, please contact me: simon.roschmann@tum.de