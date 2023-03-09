from tsaug import (
    AddNoise,
    Crop,
    Drift,
    Dropout,
    Pool,
    Quantize,
    Reverse,
)
import numpy as np


def augment(X, length, sample_rate):
    policy = [
        # Gaussian Noise
        AddNoise(loc=0, scale=(0.1, 0.2)),
        # Cutout
        Crop(
            size=(
                int(0.8 * length),
                int(1.0 * length),
            ),
            resize=int(length),
        ),
        Drift(max_drift=0.25, kind="multiplicative"),
        # Drop
        Dropout(
            p=0.05,
            fill=0,
            size=[
                # int(0.001 * sample_rate),
                int(0.01 * sample_rate),
                int(0.05 * sample_rate),
                int(0.1 * sample_rate),
            ],
        ),
        Pool(size=[2, 3, 5]),
        Quantize(n_levels=[5, 10, 15]),
        # Flip
        Reverse(),
    ]

    idx = int(np.random.choice(len(policy), size=1))
    X = policy[idx].augment(X)

    return X
