class PretrainHead(nn.Module):
    def __init__(self, n_levels, n_vars, d_model, num_patch, patch_len, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)
        self.patch_len = patch_len

        modules = []
        n_len = patch_len
        for n in range(n_levels):
            modules.append(nn.Linear(d_model, n_len))
            n_len = math.ceil(n_len * 2)

        self.mlp = nn.ModuleList(modules)

        self.mix = nn.Linear(n_levels * patch_len, patch_len)

    def forward(self, x_enc):
        """
        x: tensor [bs x nvars x num_patch x d_model]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        predicted = []

        for i in range(len(x_enc)):
            bs, n_vars, num_patch, d_model = x_enc[i].shape
            x = x_enc[i]  # [bs x nvars x num_patch x d_model]
            x = self.mlp[i](self.dropout(x))  # [bs x nvars x num_patch x patch_len]
            x = x.reshape(bs, n_vars, -1, self.patch_len)
            predicted.append(x)

        x = self.mix(torch.cat(predicted, -1))

        x = x.transpose(1, 2)  # [bs x num_patch x nvars x patch_len]
        return x


class PredictionHead(nn.Module):
    def __init__(
        self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        modules = []
        n_len = num_patch
        for n in range(n_levels):
            modules.append(nn.Linear(n_len * d_model, forecast_len))
            n_len = math.ceil(n_len / 2)

        self.mlp = nn.ModuleList(modules)

        self.mix = nn.Linear(n_levels * forecast_len, forecast_len)

    def forward(self, x_enc):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        # bs, num_patch, d_model = x.shape

        forecasted = []

        for i in range(len(x_enc)):
            x = self.flatten(x_enc[i])
            # x = self.dropout(x)
            x = self.mlp[i](x)
            forecasted.append(x)

        # y = torch.stack(forecasted, dim=-1).sum(-1)
        y = torch.cat(forecasted, dim=-1)
        y = self.mix(y)

        y = y.transpose(1, 2)

        return y


# class ResidualPredictionHead(nn.Module):
#     def __init__(
#         self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
#     ):
#         super().__init__()

#         self.flatten = nn.Flatten(start_dim=-2)
#         self.dropout = nn.Dropout(head_dropout)

#         modules = []
#         look_back_len = num_patch
#         horizon_len = forecast_len
#         for _ in range(n_levels):
#             modules.append(nn.Linear(look_back_len * d_model, horizon_len))
#             look_back_len = math.ceil(look_back_len / 2)
#             horizon_len = math.ceil(horizon_len / 2)

#         self.mlp = nn.ModuleList(modules)

#         # self.mix = nn.Linear(n_levels * forecast_len, forecast_len)

#     def forward(self, x_enc):
#         """
#         x: [bs x nvars x d_model x num_patch]
#         output: [bs x forecast_len x nvars]
#         """
#         # bs, num_patch, d_model = x.shape

#         forecasted = []

#         for i in range(len(x_enc)):
#             x = self.flatten(x_enc[i])
#             # x = self.dropout(x)
#             x = self.mlp[i](x)
#             x = torch.repeat_interleave(x, repeats=(2**i), dim=2)
#             forecasted.append(x)

#         y = torch.stack(forecasted).sum(0)

#         y = y.transpose(1, 2)

#         return y


class ResidualPredictionHead(nn.Module):
    def __init__(
        self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        modules = []
        look_back_len = num_patch
        horizon_len = forecast_len

        for n in range(n_levels):
            linear = nn.Linear(look_back_len * d_model, horizon_len)
            conv = nn.ConvTranspose1d(
                in_channels=1 if n == n_levels - 1 else 2,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding="same",
            )

            modules.append(nn.Sequential(linear, conv))

            look_back_len = math.ceil(look_back_len / 2)
            horizon_len = math.ceil(horizon_len / 2)

        # linear = nn.Linear(look_back_len * d_model, horizon_len)

        # modules.append(nn.Sequential(linear))

        self.decoder = nn.ModuleList(modules[::-1])

        # self.mix = nn.Linear(n_levels * forecast_len, forecast_len)

    def forward(self, x_enc):
        x = x_enc[-1]

        bs, n_vars, num_patch, patch_len = x.shape

        # x: [bs x nvars x num_patch x d_model]
        # x = x.reshape(bs * n_vars, -1, self.d_model)
        # x: [bs * nvars x num_patch x d_model]
        y_pred = []

        for i, (name, module) in enumerate(self.decoder.named_children()):
            for name, m_layer in module.named_children():
                # concatenate encoder outputs
                if type(m_layer) is nn.Linear:
                    x = self.flatten(x_enc[-i - 1])
                    # y = y.reshape(bs, n_vars, -1)
                    # y = torch.cat([y, x], dim=2)
                    y_lin = m_layer(x)  # linear prediction
                    # y_pred.append(y_lin)

                if type(m_layer) is UpsamplingMLP:
                    if i > 0:
                        y = torch.cat([y_lin, y], dim=2)
                    else:
                        y = y_lin

                    y = y.reshape(bs * n_vars, -1).unsqueeze(-1)
                    y = m_layer(y)
                    y = y.reshape(bs * n_vars, -1).reshape(bs, n_vars, -1)

        return y


class LowResPredictionHead(nn.Module):
    def __init__(
        self, n_levels, n_vars, d_model, num_patch, forecast_len, head_dropout
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

        n_len = math.ceil(num_patch / 2 ** (n_levels - 1))
        self.mlp = nn.Linear(n_len * d_model, forecast_len)

    def forward(self, x_enc):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        # bs, num_patch, d_model = x.shape
        x = self.flatten(x_enc[-1])
        y = self.mlp(x)
        y = y.transpose(1, 2)

        return y


# class HierarchicalPatchTSTDecoderPretraining(nn.Module):
#     def __init__(
#         self,
#         c_in,
#         num_patch,
#         patch_len,
#         num_levels,
#         num_layers,
#         num_heads,
#         d_model,
#         d_ff,
#         dropout,
#         shared_embedding=True,
#         norm="BatchNorm",
#         pre_norm=False,
#         activation="gelu",
#         pe="zeros",
#         learn_pe=False,
#         cls_token=False,
#         ch_token=False,
#         attn_dropout=0.0,
#         store_attn=False,
#         res_attention=True,
#         task=None,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.shared_embedding = shared_embedding
#         self.task = task

#         # self.W_pos = []
#         # pos_enc_len = num_patch

#         # for i in range(num_levels):
#         #     self.W_pos.append(positional_encoding(pe, learn_pe, pos_enc_len, d_model))
#         #     pos_enc_len = math.ceil(pos_enc_len / 2)

#         # residual dropout
#         self.dropout = nn.Dropout(dropout)

#         # encoder
#         dec_modules = []

#         # encoder
#         encoder = TSTEncoder(
#             num_layers=num_layers,
#             num_heads=num_heads,
#             d_model=d_model,
#             d_ff=d_ff,
#             dropout=dropout,
#             norm=norm,
#             pre_norm=pre_norm,
#             activation=activation,
#             attn_dropout=attn_dropout,
#             res_attention=res_attention,
#             store_attn=store_attn,
#         )
#         dec_modules.append(nn.Sequential(encoder))

#         for i in range(num_levels - 1):
#             dec_modules.append(
#                 nn.Sequential(
#                     UpsamplingMLP(
#                         d_model=d_model, win_size=2, norm_layer=nn.BatchNorm1d
#                     ),
#                     nn.Linear(2 * d_model, d_model),
#                     TSTEncoder(
#                         num_layers=num_layers,
#                         num_heads=num_heads,
#                         d_model=d_model,
#                         d_ff=d_ff,
#                         dropout=dropout,
#                         norm=norm,
#                         pre_norm=pre_norm,
#                         activation=activation,
#                         attn_dropout=attn_dropout,
#                         res_attention=res_attention,
#                         store_attn=store_attn,
#                     ),
#                 )
#             )

#         self.decoder = nn.Sequential(*dec_modules)

#         self.final_proj = nn.Linear(d_model, patch_len)

#     def forward(self, x_enc) -> Tensor:
#         y = x_enc[-1]

#         y_dec = []

#         bs, n_vars, num_patch, patch_len = y.shape

#         # x: [bs x nvars x num_patch x d_model]
#         y = y.reshape(bs * n_vars, -1, self.d_model)
#         # x: [bs * nvars x num_patch x d_model]

#         for i, (name, module) in enumerate(self.decoder.named_children()):
#             for name, m_layer in module.named_children():
#                 # concatenate encoder outputs
#                 if type(m_layer) is nn.Linear:
#                     x = x_enc[-i - 1]
#                     x = x.reshape(bs * n_vars, -1, self.d_model)
#                     y = torch.cat([y, x], dim=2)

#                 y = m_layer(y)

#                 if type(m_layer) is TSTEncoder:
#                     y_dec.append(
#                         y.reshape(bs, n_vars, -1, self.d_model).transpose(1, 2)
#                     )

#         pred = self.final_proj(y_dec[-1])

#         return pred


# class HierarchicalPatchTSTDecoderSupervised(nn.Module):
#     def __init__(
#         self,
#         c_in,
#         num_patch,
#         patch_len,
#         num_levels,
#         num_layers,
#         num_heads,
#         d_model,
#         d_ff,
#         dropout,
#         shared_embedding=True,
#         norm="BatchNorm",
#         pre_norm=False,
#         activation="gelu",
#         pe="zeros",
#         learn_pe=False,
#         cls_token=False,
#         ch_token=False,
#         attn_dropout=0.0,
#         store_attn=False,
#         res_attention=True,
#         task=None,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.shared_embedding = shared_embedding
#         self.task = task

#         # self.W_pos = []
#         # pos_enc_len = num_patch

#         # for i in range(num_levels):
#         #     self.W_pos.append(positional_encoding(pe, learn_pe, pos_enc_len, d_model))
#         #     pos_enc_len = math.ceil(pos_enc_len / 2)

#         # residual dropout
#         self.dropout = nn.Dropout(dropout)

#         # encoder
#         dec_modules = []

#         # encoder
#         encoder = TSTEncoder(
#             num_layers=num_layers,
#             num_heads=num_heads,
#             d_model=d_model,
#             d_ff=d_ff,
#             dropout=dropout,
#             norm=norm,
#             pre_norm=pre_norm,
#             activation=activation,
#             attn_dropout=attn_dropout,
#             res_attention=res_attention,
#             store_attn=store_attn,
#         )
#         dec_modules.append(nn.Sequential(encoder))

#         for i in range(num_levels - 1):
#             dec_modules.append(
#                 nn.Sequential(
#                     UpsamplingMLP(
#                         d_model=d_model, win_size=2, norm_layer=nn.BatchNorm1d
#                     ),
#                     nn.Linear(2 * d_model, d_model),
#                     TSTEncoder(
#                         num_layers=num_layers,
#                         num_heads=num_heads,
#                         d_model=d_model,
#                         d_ff=d_ff,
#                         dropout=dropout,
#                         norm=norm,
#                         pre_norm=pre_norm,
#                         activation=activation,
#                         attn_dropout=attn_dropout,
#                         res_attention=res_attention,
#                         store_attn=store_attn,
#                     ),
#                 )
#             )

#         self.decoder = nn.Sequential(*dec_modules)

#         self.final_proj = nn.Linear(d_model, patch_len)

#     def forward(self, x_enc) -> Tensor:
#         y = x_enc[-1]

#         y_dec = []

#         bs, n_vars, num_patch, patch_len = y.shape

#         # x: [bs x nvars x num_patch x d_model]
#         y = y.reshape(bs * n_vars, -1, self.d_model)
#         # x: [bs * nvars x num_patch x d_model]

#         for i, (name, module) in enumerate(self.decoder.named_children()):
#             for name, m_layer in module.named_children():
#                 # concatenate encoder outputs
#                 if type(m_layer) is nn.Linear:
#                     x = x_enc[-i - 1]
#                     x = x.reshape(bs * n_vars, -1, self.d_model)
#                     y = torch.cat([y, x], dim=2)

#                 y = m_layer(y)

#                 if type(m_layer) is TSTEncoder:
#                     y_dec.append(
#                         y.reshape(bs, n_vars, -1, self.d_model).transpose(1, 2)
#                     )

#         pred = self.final_proj(y_dec[-1])

#         return pred
