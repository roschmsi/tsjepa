import pandas as pd
import torch

# Parameters
epochs = 100
patience = 10
batch_size = 64 * torch.cuda.device_count()
dropout_rate = 0.2
window = 10 * 500
model_name = 'supervised_transformer'

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

debug = False
do_train = True

classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

beta = 2

class_weights = None
weights_file = 'physionet_evaluation/weights.csv'
normal_class = '426783006'