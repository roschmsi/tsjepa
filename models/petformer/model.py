# Reference: https://github.com/yuqinie98/PatchTST
import torch
from torch import Tensor, nn

from models.patch_tst.layers.encoder import TSTEncoder, TSTSignalTimeEncoder
from models.patch_tst.layers.pos_encoding import positional_encoding
from models.patch_tst.layers.revin import RevIN
from models.petformer.temporal_embedding import (
    TemporalEmbedding,
    Time2Vec,
    TimeProjectionAllInOne,
)


class PETformer(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_patch: int,
        patch_len: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_temp: int,
        d_ff: int,
        dropout: float,
        shared_embedding=True,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        attn_dropout: float = 0.0,
        res_attention: bool = False,
        store_attn: bool = False,
        task=None,
        head_dropout=0,
        individual=False,
        use_time_features=False,
        revin=False,
        temporal_attention=False,
        add_time_encoding=False,
        concat_time_encoding=False,
        input_time=False,
        patch_time=False,
    ):
        super().__init__()
        self.c_in = c_in
        self.revin = revin

        if self.revin:
            self.revin_layer = RevIN(
                num_features=c_in, affine=True, subtract_last=False
            )

        self.pred_num_patch = c_out // patch_len
        self.num_patch = num_patch + self.pred_num_patch

        self.backbone = PETformerEncoder(
            c_in=c_in,
            num_patch=self.num_patch,
            pred_num_patch=self.pred_num_patch,
            patch_len=patch_len,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_temp=d_temp,
            d_ff=d_ff,
            dropout=dropout,
            pred_len=c_out,
            shared_embedding=shared_embedding,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            pe=pe,
            learn_pe=learn_pe,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
            store_attn=store_attn,
            task=task,
            use_time_features=use_time_features,
            temporal_attention=temporal_attention,
            add_time_encoding=add_time_encoding,
            concat_time_encoding=concat_time_encoding,
            input_time=input_time,
            patch_time=patch_time,
        )

        self.projection = nn.Linear(d_model + d_temp, patch_len)

    def forward(
        self,
        z,
        X_time=None,
        y_time=None,
        padding_mask=None,
    ):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        # prepare positional encodings

        bs, num_patch, n_vars, patch_len = z.shape

        if self.revin:
            z = z.transpose(1, 2)
            z = z.reshape(bs, n_vars, num_patch * patch_len)  # bs x n_vars x seq_len
            z = z.transpose(1, 2)  # bs x seq_len x n_vars

            z = self.revin_layer(z, mode="norm")

            z = z.transpose(1, 2)
            z = z.reshape(bs, n_vars, num_patch, patch_len)
            z = z.transpose(1, 2)

        # patching should happen here

        z = self.backbone(z, X_time=X_time, y_time=y_time)
        # z: [bs x nvars x num_patch x d_model]

        z = z[:, :, -self.pred_num_patch :, :]

        z = self.projection(z)

        z = z.reshape(bs, n_vars, -1).transpose(1, 2)

        if self.revin:
            z = self.revin_layer(z, mode="denorm")

        return z


class PETformerEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        num_patch,
        pred_num_patch,
        patch_len,
        num_layers,
        num_heads,
        d_model,
        d_temp,
        d_ff,
        dropout,
        pred_len,
        shared_embedding=True,
        norm="BatchNorm",
        pre_norm=False,
        activation="gelu",
        pe="zeros",
        learn_pe=False,
        attn_dropout=0.0,
        store_attn=False,
        res_attention=False,
        task=None,
        use_time_features=False,
        temporal_attention=False,
        add_time_encoding=False,
        concat_time_encoding=False,
        input_time=False,
        patch_time=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.shared_embedding = shared_embedding
        self.task = task
        self.temporal_attention = temporal_attention
        self.add_time_encoding = add_time_encoding
        self.concat_time_encoding = concat_time_encoding
        self.input_time = input_time
        self.patch_time = patch_time

        # input encoding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(c_in):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model + d_temp)

        self.placeholder = nn.Parameter(torch.randn(1, 1, d_model))
        self.pred_num_patch = pred_num_patch

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # self.temporal_embedding = TemporalEmbedding(d_model=d_temp)
        if self.add_time_encoding:
            self.temporal_embedding = TimeProjectionAllInOne(
                d_model=d_model, patch_mode=self.patch_time
            )
        elif self.concat_time_encoding:
            self.temporal_embedding = TimeProjectionAllInOne(
                d_model=d_temp, patch_mode=self.patch_time
            )

        # encoder
        if self.temporal_attention:
            self.encoder = TSTSignalTimeEncoder(
                num_layers=num_layers,
                num_heads=num_heads,
                d_model=d_model,
                d_temp=d_temp,
                d_ff=d_ff,
                dropout=dropout,
                norm=norm,
                pre_norm=pre_norm,
                activation=activation,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
                store_attn=store_attn,
            )
        else:
            d_model = d_model + d_temp
            self.encoder = TSTEncoder(
                num_layers=num_layers,
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                norm=norm,
                pre_norm=pre_norm,
                activation=activation,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
                store_attn=store_attn,
            )

    def forward(self, x, X_time, y_time) -> Tensor:
        bs, num_patch, n_vars, patch_len = x.shape

        if self.input_time:
            if X_time is not None and y_time is not None:
                X_y_time = torch.cat([X_time, y_time], dim=1)
                temp_enc = self.temporal_embedding(X_y_time)
                temp_enc = temp_enc.unsqueeze(1)
                temp_enc = temp_enc.repeat_interleave(dim=1, repeats=n_vars)
                temp_enc = temp_enc.reshape(
                    bs * n_vars, num_patch + self.pred_num_patch, self.d_model
                )
                x = torch.cat([x, temp_enc], dim=2)

        # input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)

        # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1, 2)
        # x: [bs x nvars x num_patch x d_model]
        x = x.reshape(bs * n_vars, num_patch, self.d_model)
        # x: [bs * nvars x num_patch x d_model]

        # append placeholder tokens
        pred_tokens = torch.repeat_interleave(
            self.placeholder, dim=1, repeats=self.pred_num_patch
        )
        pred_tokens = torch.repeat_interleave(pred_tokens, dim=0, repeats=bs * n_vars)

        x = torch.cat([x, pred_tokens], dim=1)

        # add positional encoding
        # x = self.dropout(x + self.W_pos)
        # x = x + self.W_pos

        if X_time is not None and y_time is not None and self.patch_time:
            X_y_time = torch.cat([X_time, y_time], dim=1)
            temp_enc = self.temporal_embedding(X_y_time)
            # temp_enc = temp_enc.mean(dim=2)

            temp_enc = temp_enc.unsqueeze(1)
            temp_enc = temp_enc.repeat_interleave(dim=1, repeats=n_vars)

            if self.add_time_encoding:
                temp_enc = temp_enc.reshape(
                    bs * n_vars, num_patch + self.pred_num_patch, self.d_model
                )
                x = x + temp_enc
                x = x + self.W_pos
                z = self.encoder(x)

            elif self.concat_time_encoding:
                temp_enc = temp_enc.reshape(
                    bs * n_vars, num_patch + self.pred_num_patch, self.d_temp
                )
                x = torch.cat([x, temp_enc], dim=2)
                x = x + self.W_pos
                z = self.encoder(x)

            elif self.temporal_attention:
                # apply transformer encoder
                x = x + self.W_pos
                z = self.encoder(x, temp_enc)

        else:
            # x = self.dropout(x + self.W_pos)
            x = x + self.W_pos
            z = self.encoder(x)

        # for adding or appending temporal encoding
        # z = z.reshape(
        #     bs, n_vars, num_patch + self.pred_num_patch, self.d_model + self.d_temp
        # )
        z = z.reshape(
            bs, n_vars, num_patch + self.pred_num_patch, self.d_model + self.d_temp
        )
        # z: [bs x nvars x num_patch x d_model]

        # z = z.transpose(2, 3)
        # z: [bs x nvars x d_model x num_patch]

        return z
