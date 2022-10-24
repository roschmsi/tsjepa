import math
import torch
from torch.autograd import Variable
import torch.nn as nn


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    """

    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)

        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.h,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.mean(0)  # global pooling
        return out


class CTN(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, num_classes):
        super(CTN, self).__init__()

        self.encoder = nn.Sequential(  # downsampling factor = 20
            nn.Conv1d(12, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1)
        # self.fc1 = nn.Linear(d_model, deepfeat_sz)
        # self.fc2 = nn.Linear(deepfeat_sz+nb_feats+nb_demo, len(classes))
        self.fc = nn.Linear(d_model, num_classes)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.apply(_weights_init)

    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)
        z = self.encoder(x)  # encoded sequence is batch_sz x nb_ch x seq_len
        out = self.transformer(z)  # transformer output is batch_sz x d_model
        out = self.fc(out)
        return out
