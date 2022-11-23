import torch
import torch.nn as nn
import torch.nn.functional as F

from models.supervised_cnn_transformer.model import CNNEncoder
from models.supervised_fedformer.layers.AutoCorrelation import AutoCorrelationLayer
from models.supervised_fedformer.layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecomp,
    EncoderLayer,
    EncoderLayerDecomp,
    my_Layernorm,
    series_decomp,
    series_decomp_multi,
)
from models.supervised_fedformer.layers.Embed import (
    DataEmbedding_onlypos,
    DataEmbedding_OnlyToken,
    PositionalEmbedding,
)
from models.supervised_fedformer.layers.FourierCorrelation import (
    FourierBlock,
    FourierCrossAttention,
)
from models.supervised_fedformer.layers.MultiWaveletCorrelation import (
    MultiWaveletCross,
    MultiWaveletTransform,
)


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config_model, config_data):
        super(FEDformer, self).__init__()
        self.version = config_model.version
        self.mode_select = config_model.mode_select
        self.modes = config_model.modes
        self.seq_len = config_model.seq_len
        self.label_len = config_model.label_len
        self.pred_len = config_model.pred_len
        self.output_attention = config_model.output_attention

        # Decomp
        kernel_size = config_model.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config_model.enc_in,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )
        self.dec_embedding = DataEmbedding_onlypos(
            c_in=config_model.dec_in,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )

        if config_model.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=config_model.modes,
                ich=config_model.d_model,
                base=config_model.base,
                activation=config_model.cross_activation,
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=self.seq_len,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )
        # Encoder
        enc_modes = int(min(config_model.modes, config_model.seq_len // 2))
        dec_modes = int(
            min(
                config_model.modes,
                (config_model.seq_len // 2 + config_model.pred_len) // 2,
            )
        )
        print("enc_modes: {}, dec_modes: {}".format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.c_out,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.d_layers)
            ],
            norm_layer=my_Layernorm(config_model.d_model),
            projection=None,  # nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # 27 classes to predict
        self.classification_head = nn.Linear(
            config_model.d_model, config_data.num_classes
        )

        self.trend_head = nn.Linear(config_data.feat_dim, config_model.d_model)
        self.seasonal_head = nn.Linear(config_data.feat_dim, config_model.d_model)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
        )
        trend_init = self.trend_head(trend_init)
        # seasonal_init = self.seasonal_head(seasonal_init)

        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )
        # final
        dec_out = trend_part + seasonal_part

        # global pooling
        dec_out = dec_out.mean(1)

        return self.classification_head(dec_out)


class FEDformerEncoder(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config_model, config_data):
        super(FEDformerEncoder, self).__init__()
        self.version = config_model.version
        self.mode_select = config_model.mode_select
        self.modes = config_model.modes
        self.seq_len = config_model.seq_len

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config_model.enc_in,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )

        if config_model.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=self.seq_len,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )

        # Encoder
        enc_modes = int(min(config_model.modes, config_model.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(
            config_model.d_model, config_data.num_classes
        )

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = enc_out.mean(1)
        return self.classification_head(enc_out)


class CNNFEDformerEncoder(FEDformerEncoder):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config_model, config_data):
        super(CNNFEDformerEncoder, self).__init__(config_model, config_data)
        self.version = config_model.version
        self.mode_select = config_model.mode_select
        self.modes = config_model.modes
        self.seq_len = config_model.seq_len

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config_model.d_model,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )

        if config_model.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=132,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )

        # Encoder
        enc_modes = int(min(config_model.modes, config_model.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.cnn_encoder = CNNEncoder(
            feat_dim=config_model.enc_in, d_model=config_model.d_model
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(
            config_model.d_model, config_data.num_classes
        )

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

    def __init__(self, config_model, config_data):
        super(CNNTimeFreqEncoder, self).__init__()
        self.version = config_model.version
        self.mode_select = config_model.mode_select
        self.modes = config_model.modes
        self.seq_len = config_model.seq_len

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config_model.d_model,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )

        if config_model.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=132,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )

        # Encoder
        enc_modes = int(min(config_model.modes, config_model.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.cnn_encoder = CNNEncoder(
            feat_dim=config_model.enc_in, d_model=config_model.d_model
        )

        self.freq_transformer = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        time_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config_model.d_model,
            nhead=config_model.num_heads,
            dim_feedforward=config_model.d_ff,
            dropout=config_model.dropout,
        )
        self.time_transformer = nn.TransformerEncoder(
            time_transformer_layer, config_model.e_layers
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(
            2 * config_model.d_model, config_data.num_classes
        )

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.cnn_encoder(x_enc)
        enc_out = self.enc_embedding(enc_out)
        freq_enc_out, _ = self.freq_transformer(enc_out, attn_mask=enc_self_mask)
        time_enc_out = self.time_transformer(enc_out.transpose(0, 1))
        enc_out = torch.cat([freq_enc_out, time_enc_out.transpose(0, 1)], dim=2)
        enc_out = enc_out.mean(1)

        return self.classification_head(enc_out)


class FEDformerEncoderDecomp(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config_model, config_data):
        super(FEDformerEncoderDecomp, self).__init__()
        self.version = config_model.version
        self.mode_select = config_model.mode_select
        self.modes = config_model.modes
        self.seq_len = config_model.seq_len

        # Decomp
        kernel_size = config_model.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config_model.d_model,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )

        if config_model.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=self.seq_len,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )
        # Encoder
        enc_modes = int(min(config_model.modes, config_model.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.encoder = EncoderDecomp(
            [
                EncoderLayerDecomp(
                    AutoCorrelationLayer(
                        encoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )

        # 27 classes to predict
        self.classification_head = nn.Linear(
            config_model.d_model, config_data.num_classes
        )

        self.trend_head = nn.Linear(config_data.feat_dim, config_model.d_model)
        self.seasonal_head = nn.Linear(config_data.feat_dim, config_model.d_model)

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


class CNNTimeFreqEncoderDecomp(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, config_model, config_data):
        super(CNNTimeFreqEncoderDecomp, self).__init__()
        self.version = config_model.version
        self.mode_select = config_model.mode_select
        self.modes = config_model.modes
        self.seq_len = config_model.seq_len

        kernel_size = config_model.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(
            c_in=config_model.d_model,
            d_model=config_model.d_model,
            dropout=config_model.dropout,
        )

        if config_model.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config_model.d_model, L=config_model.L, base=config_model.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config_model.d_model,
                out_channels=config_model.d_model,
                seq_len=132,
                modes=config_model.modes,
                mode_select_method=config_model.mode_select,
            )

        # Encoder
        enc_modes = int(min(config_model.modes, config_model.seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.cnn_encoder = CNNEncoder(
            feat_dim=config_model.enc_in, d_model=config_model.d_model
        )

        self.freq_transformer = EncoderDecomp(
            [
                EncoderLayerDecomp(
                    AutoCorrelationLayer(
                        encoder_self_att, config_model.d_model, config_model.num_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            # norm_layer=my_Layernorm(config_model.d_model),
        )
        time_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config_model.d_model,
            nhead=config_model.num_heads,
            dim_feedforward=config_model.d_ff,
            dropout=config_model.dropout,
        )
        self.time_transformer = nn.TransformerEncoder(
            time_transformer_layer, config_model.e_layers
        )
        # 27 classes to predict
        self.classification_head = nn.Linear(
            2 * config_model.d_model, config_data.num_classes
        )

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
