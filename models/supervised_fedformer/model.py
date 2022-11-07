import torch
import torch.nn as nn
import torch.nn.functional as F
from models.supervised_fedformer.layers.Embed import DataEmbedding_OnlyToken
from models.supervised_fedformer.layers.AutoCorrelation import AutoCorrelationLayer
from models.supervised_fedformer.layers.FourierCorrelation import (
    FourierBlock,
    FourierCrossAttention,
)
from models.supervised_fedformer.layers.MultiWaveletCorrelation import (
    MultiWaveletCross,
    MultiWaveletTransform,
)
from models.supervised_fedformer.layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
    series_decomp_multi,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.enc_embedding = DataEmbedding_OnlyToken(
            config_model.enc_in,
            config_model.d_model,
            config_model.dropout,
        )
        self.dec_embedding = DataEmbedding_OnlyToken(
            config_model.dec_in,
            config_model.d_model,
            config_model.dropout,
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
                        encoder_self_att, config_model.d_model, config_model.n_heads
                    ),
                    config_model.d_model,
                    config_model.d_ff,
                    moving_avg=config_model.moving_avg,
                    dropout=config_model.dropout,
                    activation=config_model.activation,
                )
                for _ in range(config_model.e_layers)
            ],
            norm_layer=my_Layernorm(config_model.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att, config_model.d_model, config_model.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, config_model.d_model, config_model.n_heads
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


class FedformerEncoder(FEDformer):
    def __init__(self, config_model, config_data):
        super(FedformerEncoder, self).__init__(config_model, config_data)

        self.classification_head = nn.Linear(
            config_model.d_model, config_data.num_classes
        )

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # global pooling
        enc_out = enc_out.mean(1)

        return self.classification_head(enc_out)
