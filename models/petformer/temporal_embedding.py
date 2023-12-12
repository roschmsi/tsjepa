import torch.nn as nn
import torch.nn.functional as F
import torch


# inspired by informer
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 31
        month_size = 12

        if freq == "t":
            self.minute_embed = nn.Embedding(minute_size, d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


# newly invented
class Time2Vec(nn.Module):
    def __init__(self, individual=False):
        super(Time2Vec, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 31
        month_size = 12

        self.individual = individual

    def forward(self, x):
        x = x.long()

        # minute_x = (
        #     self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        # )
        hour_x = F.one_hot(x[:, :, 3], num_classes=24).to(torch.float32)
        weekday_x = F.one_hot(x[:, :, 2], num_classes=7).to(torch.float32)
        day_x = F.one_hot(x[:, :, 1], num_classes=31).to(torch.float32)
        month_x = F.one_hot(x[:, :, 0], num_classes=12).to(torch.float32)

        # timestamp = torch.cat([hour_x, weekday_x, day_x, month_x], dim=-1).to(
        #     torch.float32
        # )
        if self.individual:
            hour_x = F.normalize(hour_x.mean(dim=2), p=2, dim=-1)
            weekday_x = F.normalize(weekday_x.mean(dim=2), p=2, dim=-1)
            day_x = F.normalize(day_x.mean(dim=2), p=2, dim=-1)
            month_x = F.normalize(month_x.mean(dim=2), p=2, dim=-1)
            timestamp = torch.cat([hour_x, weekday_x, day_x, month_x], dim=-1)
            return timestamp
        else:
            # hour_x missing here
            timestamp = torch.cat([hour_x, weekday_x, day_x, month_x], dim=-1)
            # mean over patch, then normalize for scalar product
            timestamp = timestamp.mean(dim=2)
            timestamp = F.normalize(timestamp.float(), p=2, dim=-1)
            return timestamp


# newly invented
class TimeProjectionAllInOne(nn.Module):
    def __init__(self, d_model, patch_mode, individual=False):
        super(TimeProjectionAllInOne, self).__init__()

        self.projection = nn.Linear(4 + 24 + 7 + 31 + 12, d_model)
        self.patch_mode = patch_mode

    def forward(self, x):
        x = x.long()

        minute_x = F.one_hot(x[:, :, 4], num_classes=4).to(torch.float32)
        hour_x = F.one_hot(x[:, :, 3], num_classes=24).to(torch.float32)
        weekday_x = F.one_hot(x[:, :, 2], num_classes=7).to(torch.float32)
        day_x = F.one_hot(x[:, :, 1], num_classes=31).to(torch.float32)
        month_x = F.one_hot(x[:, :, 0], num_classes=12).to(torch.float32)

        timestamp = torch.cat([minute_x, hour_x, weekday_x, day_x, month_x], dim=-1)

        timestamp = self.projection(timestamp)

        if self.patch_mode:
            timestamp = timestamp.mean(dim=2)

        # normalize
        timestamp = timestamp - timestamp.mean()
        timestamp = timestamp / timestamp.std()

        return timestamp
