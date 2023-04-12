import torch
from torch import nn

from models.patch_tst.layers.encoder import TSTEncoder
from models.patch_tst.layers.pos_encoding import positional_encoding


class PredictionHead(nn.Module):
    def __init__(
        self,
        individual,
        n_vars,
        d_model,
        num_patch,
        forecast_len,
        head_dropout=0,
        flatten=False,
    ):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch
        self.forecast_len = forecast_len

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, forecast_len * n_vars)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x num_patch x d_model]
        output: [bs x forecast_len x nvars]
        """
        x = self.flatten(x)
        # x: [bs x num_patch * d_model]
        x = self.dropout(x)
        x = self.linear(x)
        # x: [bs x forecast_len * n_vars]
        x = x.reshape(-1, self.forecast_len, self.n_vars)
        # x: [bs x forecast_len x nvars]

        return x


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, nvars, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, nvars * patch_len)
        self.nvars = nvars
        self.patch_len = patch_len

    def forward(self, x):
        """
        x: tensor [bs x num_patch x d_model]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        bs, num_patch, d_model = x.shape
        x = self.linear(self.dropout(x))
        # x: [bs x num_patch x n_vars * patch_len]
        x = torch.reshape(x, (bs, num_patch, self.nvars, self.patch_len))
        # x: [bs x num_patch x n_vars x patch_len]

        return x


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout, n_patch=None):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x num_patch x d_model]
        output: [bs x n_classes]
        """
        # extract class token
        x = x[:, 0, :]
        # x: [bs x d_model]
        x = self.dropout(x)
        y = self.linear(x)
        # y: [bs x n_classes]

        return y


class PatchTransformerT(nn.Module):
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
        d_ff: int,
        dropout: float,
        norm: str = "BatchNorm",
        pre_norm: bool = False,
        activation: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = False,
        cls_token: bool = False,
        attn_dropout: float = 0.0,
        res_attention: bool = True,
        store_attn: bool = False,
        task=None,
        head_dropout=0,
        individual=False,
    ):
        super().__init__()

        self.backbone = PatchTSTEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            norm=norm,
            pre_norm=pre_norm,
            activation=activation,
            pe=pe,
            learn_pe=learn_pe,
            cls_token=cls_token,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
            store_attn=store_attn,
            task=task,
        )

        if task == "pretraining":
            self.head = PretrainHead(
                d_model=d_model,
                patch_len=patch_len,
                nvars=c_in,
                dropout=head_dropout,
            )
        elif task == "forecasting":
            self.head = PredictionHead(
                individual=individual,
                n_vars=c_in,
                d_model=d_model,
                num_patch=num_patch,
                forecast_len=c_out,
                head_dropout=head_dropout,
            )
        elif task == "classification":
            self.head = ClassificationHead(
                n_vars=c_in,
                d_model=d_model,
                n_classes=c_out,
                head_dropout=head_dropout,
            )

    def forward(self, z, padding_mask=None):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        z = self.backbone(z.float())  # z: [bs x nvars x d_model x num_patch]
        z = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        num_layers,
        num_heads,
        d_model,
        d_ff,
        dropout,
        norm="BatchNorm",
        pre_norm=False,
        activation="gelu",
        pe="zeros",
        learn_pe=False,
        cls_token=False,
        attn_dropout=0.0,
        store_attn=False,
        res_attention=True,
        task=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.task = task

        # input encoding
        self.W_P = nn.Linear(patch_len * c_in, d_model)

        # class tokens
        self.cls_token = nn.Parameter(torch.zeros(1, d_model)) if cls_token else None

        # positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # residual dropout
        self.dropout = nn.Dropout(dropout)

        # encoder
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

    def forward(self, x, padding_mask=None):
        bs, num_patch, n_vars, patch_len = x.shape

        x = x.reshape(bs, num_patch, n_vars * patch_len)

        # input encoding
        x = self.W_P(x)
        # x: [bs x num_patch x d_model]

        # add positional encoding
        x = self.dropout(x + self.W_pos)

        # append class token at start
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # apply transformer encoder
        z = self.encoder(x)
        # z: [bs x num_patch x d_model]

        if self.task != "classification" and self.cls_token is not None:
            z = z[:, 1:, :]
        # z: [bs x num_patch x d_model]

        return z
