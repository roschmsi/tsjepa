# Reference: https://github.com/seitalab/RandECG

import numpy as np
from scipy import signal as scipy_signal
import biosppy

FREQ = 50
filter_order = 10


def random_scale(X, Mmax, Mmin):
    """
    Randomly scale given batch.

    rand (0 - 1) * (max_val - min_val) -> 0 ~ max_val - min_val
    -> + min_val -> min_val ~ max_val

    Args:
        X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        Mmax: Maximum value of scaling parameter.
        Mmin: Minimum value of scaling parameter.
    Returns:
        scaledX (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
    """
    scaledX = X.copy()
    num_batch = X.shape[0]
    scaler = np.random.rand(num_batch) * (Mmax - Mmin) + Mmin
    scaledX *= scaler[:, np.newaxis, np.newaxis]
    return scaledX


def random_flip(X, M):
    """
    Randomly flip signal based on value of M.
    (if M == 0.2: 20% of samples are flipped.)

    Args:
        X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        M (float): Maximum value for flipping (Value between 0 - 1.)
    Returns:
        Xflipped (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
    """
    Xflipped = X.copy()
    num_batch = X.shape[0]
    flipper = np.random.rand(num_batch) < M
    # True -> 1 * (-2) -> -2 + 1 -> -1
    # False -> 0 * (-2) -> 0 + 1 -> 1
    flipper = flipper.astype("float") * -2 + 1
    Xflipped *= flipper[:, np.newaxis, np.newaxis]
    return Xflipped


def random_drop(X, M):
    """
    Randomly mask signal based on value of M.

    Args:
        X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        M (float): Value between 0 - 1.
    Returns:
        Xdropped (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
    """
    Xdropped = X.copy()
    seqlen = X.shape[-1]
    # if val > M: 1 -> original signal
    # if val < M: 0 -> masked
    drop_loc = np.random.rand(seqlen) > M
    Xdropped *= drop_loc[np.newaxis, np.newaxis, :]
    return Xdropped


def random_cutout(X, M):
    """
    Randomly cutout signal.

    Args:
        X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        M (float): Maximum width ratio to cutout (value between 0 - 1.)
    Returns:
        Xcut (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
    """
    Xcut = X.copy()
    signal_length = X.shape[2]

    w_ratio = np.random.rand() * M  # Random value between 0 - M
    width = int(signal_length * w_ratio)
    start = int(np.random.rand() * (signal_length - width))
    for w in range(width):
        Xcut[:, :, start + w] = 0
    return Xcut


def random_shift(X, M):
    """

    Args:
        X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        M (float): Maximum width ratio to shift (value between 0 - 1.)
                  (shift direction is randomly decided)
    Returns:
        X_shift (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
    """
    X_shift = X.copy()
    signal_length = X.shape[2]
    # random value between -M to M.
    w_ratio = (np.random.rand() - 0.5) * 2 * M

    shift = int(signal_length * w_ratio)
    for i in range(len(X)):
        pad_size = np.zeros([X.shape[1], abs(shift)])
        if shift > 0:
            # shift direction: ->
            X_shift[i] = np.concatenate([pad_size, X[i, :, : -abs(shift)]], axis=1)
        elif shift < 0:
            # shift direction: <-
            X_shift[i] = np.concatenate([X[i, :, abs(shift) :], pad_size], axis=1)
    return X_shift


def sine_noise(X, M, F):
    """
    Args:
        X: [batchsize, num_channel, sequence_length]
        M (float): Value for amplitude (value between 0 - 1).
        F (float): Value for frequency
    Returns:
        X_sine:
    """
    seqlen = X.shape[2]
    duration = seqlen / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * F, seqlen)

    X_sine = X.copy()
    X_sine += M * np.sin(steps)
    return X_sine


def square_noise(X, M, F):
    """
    Args:
        X: [batchsize, num_channel, sequence_length]
        M (float): Value for amplitude (value between 0 - 1).
        F (float): Value for frequency
    Returns:
        X_square:
    """
    seqlen = X.shape[2]
    duration = seqlen / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * F, seqlen)

    X_square = X.copy()
    X_square += M * scipy_signal.square(steps)
    return X_square


def white_noise(X, M):
    """
    Args:
        X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        M (float): Amplitude of white noise (Value between 0 - 1.)
    Returns:
        X_wn (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
    """
    X_wn = X.copy()

    amp = np.random.rand() * M  # Random value between 0 - M
    # gaussian_noise centered to 0
    white_noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2])
    X_wn += white_noise * amp
    return X_wn


def sine_noise_partial(X, M, F):
    """
    Args:
        X: [batchsize, num_channel, sequence_length]
        M: value between 0 - 1.
        F (float):
    Returns:
        X_sine_p:
    """
    X_sine_p = X.copy()
    signal_length = X.shape[2]

    duration = signal_length / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * F, signal_length)
    sine_curve = np.sin(steps)

    w_ratio = np.random.rand() * M  # Random value between 0 - M
    width = int(signal_length * w_ratio)
    start = int(np.random.rand() * (signal_length - width))
    for w in range(width):
        X_sine_p[:, :, start + w] += sine_curve[np.newaxis, np.newaxis, w]
    return X_sine_p


def square_noise_partial(X, M, F):
    """
    Args:
        X: [batchsize, num_channel, sequence_length]
        M:
        fs (float):
    Returns:
        X_square_p:
    """
    X_square_p = X.copy()
    signal_length = X.shape[2]

    duration = signal_length / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * F, signal_length)
    square_pulse = scipy_signal.square(steps)

    w_ratio = np.random.rand() * M  # Random value between 0 - M
    width = int(signal_length * w_ratio)
    start = int(np.random.rand() * (signal_length - width))
    for w in range(width):
        X_square_p[:, :, start + w] += square_pulse[np.newaxis, np.newaxis, w]
    return X_square_p


def white_noise_partial(X, M):
    """
    Args:
        X: [batchsize, num_channel, sequence_length]
        M: Magnitude of partial noise, corresponding to width of sample.
    Returns:
        X_wnp:
    """
    X_wnp = X.copy()
    signal_length = X.shape[2]

    w_ratio = np.random.rand() * M  # Random value between 0 - M
    width = int(signal_length * w_ratio)
    start = int(np.random.rand() * (signal_length - width))
    white_noise = np.random.randn(X.shape[0], X.shape[1], width)

    for w in range(width):
        X_wnp[:, :, start + w] += white_noise[:, :, w]
    return X_wnp


def apply_FIR_low(X, Mmax, Mmin):
    """
    Args:
        X:
        M:
    Returns:
        filtered_X:
    """
    f_ratio = np.random.rand() * (Mmax - Mmin) + Mmin

    freq = max(1, min(int(f_ratio * FREQ), FREQ / 2 - 1))
    filtered_X, _, _ = biosppy.tools.filter_signal(
        signal=X,
        ftype="FIR",
        band="lowpass",
        order=filter_order,
        frequency=freq,
        sampling_rate=FREQ,
    )
    return filtered_X
