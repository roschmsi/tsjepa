import pandas as pd
import torch

# Parameters
debug = False
patience = 10
batch_size = 16 * torch.cuda.device_count()
window = 15*500
dropout_rate = 0.2
deepfeat_sz = 64
padding = 'zero' # 'zero', 'qrs', or 'none'
fs = 500
filter_bandwidth = [3, 45]
model_name = 'ctn'

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

do_train = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

thrs_per_class = False
class_weights = None

classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

# Load all features dataframe
data_df = pd.read_csv('records_stratified_10_folds_v2.csv', index_col=0)

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead2idx = dict(zip(leads, range(len(leads))))

beta = 2
num_classes = len(classes)

weights_file = 'eval/weights.csv'
normal_class = '426783006'
normal_index = classes.index(normal_class)
normal_lbl = [0. if i != normal_index else 1. for i in range(num_classes)]
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]