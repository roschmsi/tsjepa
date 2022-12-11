from tsaug import (
    AddNoise,
    Crop,
    Drift,
    Dropout,
    Pool,
    Quantize,
    Reverse,
    TimeWarp,
)
import numpy as np


class Augmenter:
    def __init__(self, sample_rate, window, prob=0.8):
        self.policy = [
            AddNoise(loc=0, scale=(0.1, 0.5)),
            Crop(
                size=(
                    int(0.25 * sample_rate * window),
                    int(0.75 * sample_rate * window),
                ),
                resize=int(sample_rate * window),
            ),
            Drift(max_drift=(0.1, 0.5)),
            Dropout(
                p=0.1,
                fill=0,
                size=[
                    # int(0.001 * sample_rate),
                    int(0.01 * sample_rate),
                    int(0.05 * sample_rate),
                    int(0.1 * sample_rate),
                ],
            ),
            Pool(size=(2, 5)),
            Quantize(n_levels=[10, 20, 30]),
            Reverse(),
            TimeWarp(n_speed_change=5, max_speed_ratio=(2, 5)),
        ]
        self.augmentation_prob = prob

    # def augment(self, X, sample_length):
    #     policy = [
    #         AddNoise(loc=0, scale=(0.1, 0.5)),
    #         Crop(
    #             size=(
    #                 int(0.25 * sample_rate * window),
    #                 int(0.75 * sample_rate * window),
    #             ),
    #             resize=int(sample_rate * window),
    #         ),
    #         Drift(max_drift=(0.1, 0.5)),
    #         Dropout(
    #             p=0.1,
    #             fill=0,
    #             size=[
    #                 # int(0.001 * sample_rate),
    #                 int(0.01 * sample_rate),
    #                 int(0.05 * sample_rate),
    #                 int(0.1 * sample_rate),
    #             ],
    #         ),
    #         Pool(size=(2, 5)),
    #         Quantize(n_levels=[10, 20, 30]),
    #         Reverse(),
    #         TimeWarp(n_speed_change=5, max_speed_ratio=(2, 5)),
    #     ]
    #     if random.random() < self.augmentation_prob:
    #         idx = int(np.random.choice(len(self.policy), size=1))
    #         X = self.policy[idx].augment(X)
    #     return X


def augment(X, length, sample_rate):
    policy = [
        AddNoise(loc=0, scale=(0.1, 0.5)),
        Crop(
            size=(
                int(0.5 * length),
                int(1.0 * length),
            ),
            resize=int(length),
        ),
        Drift(max_drift=0.5, kind="multiplicative"),
        Dropout(
            p=0.1,
            fill=0,
            size=[
                # int(0.001 * sample_rate),
                int(0.01 * sample_rate),
                int(0.05 * sample_rate),
                int(0.1 * sample_rate),
            ],
        ),
        Pool(size=[2, 3, 5]),
        Quantize(n_levels=[10, 20, 30]),
        Reverse(),
        TimeWarp(n_speed_change=3, max_speed_ratio=(2, 3)),
    ]

    idx = int(np.random.choice(len(policy), size=1))
    X = policy[idx].augment(X)

    return X
