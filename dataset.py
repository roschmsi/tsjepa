import numpy as np
from torch.utils.data import Dataset

from scipy.io import loadmat
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal

from utils import *


class ECGDataset(Dataset):
    def __init__(self, df, window, nb_windows, src_path):
        ''' Return randome window length segments from ecg signal, pad if window is too large
            df: trn_df, val_df or tst_df
            window: ecg window length e.g 2500 (5 seconds)
            nb_windows: number of windows to sample from record
        '''
        self.df = df
        self.window = window
        self.nb_windows = nb_windows
        self.src_path = src_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path/(row.Patient + '.hea'))
        data, hdr = load_challenge_data(filename)
        seq_len = data.shape[-1] # get the length of the ecg sequence
        
        # Apply band pass filter
        if filter_bandwidth is not None:
            data = apply_filter(data, filter_bandwidth)
        
        data = normalize(data)
        lbl = row[classes].values.astype(np.int)
        
        # Add just enough padding to allow window
        pad = np.abs(np.min(seq_len - self.window, 0))
        if pad > 0:
            data = np.pad(data, ((0,0),(0,pad+1)))
            seq_len = data.shape[-1] # get the new length of the ecg sequence
        
        starts = np.random.randint(seq_len - self.window + 1, size=self.nb_windows) # get start indices of ecg segment        
        ecg_segs = np.array([data[:,start:start+self.window] for start in starts])
        return ecg_segs, lbl, filename       

def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()    
    sampling_rate = int(header[0].split()[2])    
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    
    # Standardize sampling rate
    if sampling_rate > fs:
        recording = decimate(recording, int(sampling_rate / fs))
    elif sampling_rate < fs:
        recording = resample(recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1)
    
    return recording, header

def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def apply_filter(signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal
        
            
    
