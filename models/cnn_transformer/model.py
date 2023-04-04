import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.transformer.model import (
    FixedPositionalEncoding,
    LearnablePositionalEncoding,
)


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

    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        max_seq_len,
        learn_pe=False,
    ):
        super(Transformer, self).__init__()
        # self.pe = PositionalEncoding(d_model, dropout=0.1)
        if learn_pe:
            self.pe = LearnablePositionalEncoding(
                d_model=d_model, dropout=dropout, max_len=max_seq_len
            )
        else:
            self.pe = FixedPositionalEncoding(
                d_model=d_model, dropout=dropout, max_len=max_seq_len, scale_factor=1.0
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, padding_mask=None):
        # x: bs x d_model x seq_len
        # out: bs x d_model x seq_len
        out = x.permute(2, 0, 1)
        out = self.pe(out)
        out = self.transformer_encoder(out)
        out = out.permute(1, 2, 0)
        return out


class CNNEncoder(nn.Module):
    def __init__(
        self,
        feat_dim,
        d_model,
        num_cnn,
    ):
        super(CNNEncoder, self).__init__()

        modules = [
            nn.Conv1d(
                feat_dim, d_model, kernel_size=11, stride=3, padding=0, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                d_model, d_model, kernel_size=11, stride=3, padding=0, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        ]
        modules = modules[: (3 * num_cnn)]
        self.encoder = nn.Sequential(*modules)

    def forward(self, x, padding_mask=None):
        # x: bs x feat_dim x seq_len
        # out: bs x d_model x seq_len
        out = self.encoder(x)
        return out


class CNNTransformer(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        num_classes,
        max_seq_len,
        num_cnn,
        cls_token,
        learn_pe,
    ):
        super(CNNTransformer, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, d_model, 1)) if cls_token else None

        self.encoder = CNNEncoder(feat_dim=feat_dim, d_model=d_model, num_cnn=num_cnn)
        self.transformer = Transformer(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            learn_pe=learn_pe,
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        # x: bs x d_model x seq_len
        x = x.transpose(1, 2)
        out = self.encoder(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(out.shape[0], -1, -1)
            out = torch.cat((cls_token, out), dim=2)

        out = self.transformer(out)  # out: bs x d_model x seq_len

        if self.cls_token is not None:
            out = out[:, :, 0]
        else:
            out = out.mean(-1)

        out = self.fc(out)
        return out


class CNNClassifier(nn.Module):
    def __init__(
        self,
        feat_dim,
        d_model,
        num_classes,
        num_cnn,
    ):
        super(CNNClassifier, self).__init__()

        self.encoder = CNNEncoder(feat_dim=feat_dim, d_model=d_model, num_cnn=num_cnn)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        # x: bs x d_model x seq_len
        x = x.transpose(1, 2)
        out = self.encoder(x)  # out: bs x d_model x seq_len
        out = out.mean(-1)
        out = self.fc(out)
        return out
