# Reference: https://github.com/MAZiqing/FEDformer

import torch
import torch.nn as nn

from models.fedformer.layers.AutoCorrelation import AutoCorrelationLayer
from models.fedformer.layers.Autoformer_EncDec import (
    Encoder,
    EncoderDecomp,
    EncoderLayer,
    EncoderLayerDecomp,
    series_decomp,
    series_decomp_multi,
)
from models.fedformer.layers.Embed import DataEmbedding_onlypos
from models.fedformer.layers.FourierCorrelation import FourierBlock
from models.fedformer.layers.MultiWaveletCorrelation import MultiWaveletTransform


class FEDformerEncoder(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config):
        super(FEDformerEncoder, self).__init__()
        self.version = config.version
        self.mode_select = config.mode_select
        self.modes = config.modes
        self.seq_len = config.window * config.fs

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config.feat_dim,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        if config.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config.d_model, L=config.L, base=config.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len=self.seq_len,
                modes=config.modes,
                mode_select_method=config.mode_select,
            )

        # Encoder
        enc_modes = int(min(config.modes, self.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config.d_model, config.num_heads
                    ),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = enc_out.mean(1)
        return self.classification_head(enc_out)


class CNNFEDformerEncoder(FEDformerEncoder):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config):
        super(CNNFEDformerEncoder, self).__init__(config, config)
        self.version = config.version
        self.mode_select = config.mode_select
        self.modes = config.modes
        self.seq_len = config.window * config.fs

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config.d_model,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        if config.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config.d_model, L=config.L, base=config.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len=132,
                modes=config.modes,
                mode_select_method=config.mode_select,
            )

        # Encoder
        enc_modes = int(min(config.modes, self.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.cnn_encoder = CNNEncoderLayer(
            feat_dim=config.feat_dim, d_model=config.d_model
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config.d_model, config.num_heads
                    ),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.cnn_encoder(x_enc)
        enc_out = self.enc_embedding(enc_out)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = enc_out.mean(1)
        return self.classification_head(enc_out)


class CNNTimeFreqEncoder(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config):
        super(CNNTimeFreqEncoder, self).__init__()
        self.version = config.version
        self.mode_select = config.mode_select
        self.modes = config.modes
        self.seq_len = config.window * config.fs

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config.d_model,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        if config.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config.d_model, L=config.L, base=config.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len=132,
                modes=config.modes,
                mode_select_method=config.mode_select,
            )

        # Encoder
        enc_modes = int(min(config.modes, self.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.cnn_encoder = CNNEncoderLayer(
            feat_dim=config.feat_dim, d_model=config.d_model
        )

        self.freq_transformer = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config.d_model, config.num_heads
                    ),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        time_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )
        self.time_transformer = nn.TransformerEncoder(
            time_transformer_layer, config.num_layers
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(2 * config.d_model, config.num_classes)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.cnn_encoder(x_enc)
        enc_out = self.enc_embedding(enc_out)
        freq_enc_out, _ = self.freq_transformer(enc_out, attn_mask=enc_self_mask)
        time_enc_out = self.time_transformer(enc_out.transpose(0, 1))
        enc_out = torch.cat([freq_enc_out, time_enc_out.transpose(0, 1)], dim=2)
        enc_out = enc_out.mean(1)

        return self.classification_head(enc_out)


class DecompFEDformerEncoder(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        activation,
        version,
        modes,
        mode_select,
        seq_len,
        moving_avg,
        feat_dim,
        num_classes,
        L=None,
        base=None,
    ):
        super(DecompFEDformerEncoder, self).__init__()
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=d_model,
            d_model=d_model,
            dropout=dropout,
        )

        if version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
        else:
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=self.seq_len,
                modes=modes,
                mode_select_method=mode_select,
            )
        # Encoder
        enc_modes = int(min(modes, self.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.encoder = EncoderDecomp(
            [
                EncoderLayerDecomp(
                    AutoCorrelationLayer(encoder_self_att, d_model, num_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )

        # 27 classes to predict
        self.classification_head = nn.Linear(d_model, num_classes)

        self.trend_head = nn.Linear(feat_dim, d_model)
        self.seasonal_head = nn.Linear(feat_dim, d_model)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.seq_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        # trend_init = torch.cat([trend_init, mean], dim=1)

        trend_init = self.trend_head(trend_init)
        seasonal_init = self.seasonal_head(seasonal_init)

        # enc
        enc_out = self.enc_embedding(seasonal_init)  # or enc_in as input
        seasonal_part, trend_part = self.encoder(
            enc_out, trend=trend_init, attn_mask=enc_self_mask
        )

        # final
        enc_out = trend_part + seasonal_part

        # global pooling
        enc_out = enc_out.mean(1)

        return self.classification_head(enc_out)


class CNNDecompTimeFreqEncoder(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config):
        super(CNNDecompTimeFreqEncoder, self).__init__()
        self.version = config.version
        self.mode_select = config.mode_select
        self.modes = config.modes
        self.seq_len = config.window * config.fs

        kernel_size = config.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config.d_model,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        if config.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config.d_model, L=config.L, base=config.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len=132,
                modes=config.modes,
                mode_select_method=config.mode_select,
            )

        # Encoder
        enc_modes = int(min(config.modes, self.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.cnn_encoder = CNNEncoderLayer(
            feat_dim=config.feat_dim, d_model=config.d_model
        )

        self.freq_transformer = EncoderDecomp(
            [
                EncoderLayerDecomp(
                    AutoCorrelationLayer(
                        encoder_self_att, config.d_model, config.num_heads
                    ),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        time_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )
        self.time_transformer = nn.TransformerEncoder(
            time_transformer_layer, config.num_layers
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(2 * config.d_model, config.num_classes)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.cnn_encoder(x_enc)

        # time transformer
        time_enc_out = self.enc_embedding(enc_out)
        time_enc_out = self.time_transformer(time_enc_out.transpose(0, 1))

        # frequency transformer
        seasonal_init, trend_init = self.decomp(x_enc)
        freq_enc_out = self.enc_embedding(seasonal_init)  # or enc_in as input
        seasonal_part, trend_part = self.freq_transformer(
            freq_enc_out, trend=trend_init, attn_mask=enc_self_mask
        )
        freq_enc_out = trend_part + seasonal_part

        enc_out = torch.cat([freq_enc_out, time_enc_out.transpose(0, 1)], dim=2)
        enc_out = enc_out.mean(1)

        return self.classification_head(enc_out)
