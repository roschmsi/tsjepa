# Reference: https://github.com/seitalab/RandECG

from rand_ecg.augment_functions import *
import numpy as np

# params
Mstep = 20
FREQ = 50
# `M` will be selected based on magnitude
scale_range_Mmax = (1, 4)  # Scaling range Max
scale_range_Mmin = (0.25, 1)  # Scaling range Min
flip_range_M = (0, 1)  # Probability of flipping
drop_range_M = (0, 0.4)  # Probability of dropping
cutout_range_M = (0, 0.4)  # Width of cutout
shift_range_M = (0, 1.0)  # Maximum ratio of shifting

white_noise_range_M = (0, 0.02)  # Amplitude of white noise
sine_noise_range_M = (0, 1.0)  # Amplitude of sine noise
square_noise_range_M = (0, 0.02)  # Amplitude of square pulse

partial_white_noise_range_M = (0, 0.1)  # Width of white noise applied region
partial_sine_noise_range_M = (0, 0.1)  # Width of sine noise applied region
partial_square_noise_range_M = (0, 1.0)  # Width of square pulse applied region
fir_low_range_M = (0.1, 0.4999)  # Filter frequency range (lowpass)

sine_noise_range_F = (0.001, 0.02)  # Frequency of sine noise
square_noise_range_F = (0.001, 0.1)  # Frequency of square pulse
partial_sine_noise_range_F = (0.1, 1)  # Frequency of partial sine noise
partial_square_noise_range_F = (0.02, 1)  # Frequency of partial square pulse


augment_choices = {
    "rand": [
        "scale",
        "flip",
        "drop",
        "cutout",
        "shift",
        "sine",
        "square",
        "wn",
        "sine_p",
        "square_p",
        "wn_p",
        # "fir_l",
        "none",
    ],
}


def rand_val(*args):
    min_val, max_val = args[0], args[1]
    return (max_val - min_val) * np.random.rand() + min_val


def step_val(M, *args):
    min_val, max_val = args[0], args[1]
    Mval = float(M / Mstep) * (max_val - min_val) + min_val
    return Mval


def step_val_rev(M, *args):
    min_val, max_val = args[0], args[1]
    Mval = max_val - ((max_val - min_val) * float(M / Mstep))
    return Mval


def get_M_and_N(augment_type):
    head, M, N = augment_type.split("-")
    assert head in augment_choices.keys()
    assert M[0] == "M"
    assert N[0] == "N"
    M = int(M[1:])
    N = int(N[1:])
    augment_operations = augment_choices[head]
    return M, N, augment_operations


def randaug(X_batch, augment_type):
    """
    Apply random augmentation.

    Args:
        X_batch:
        augment_type: String of `<rand_type>-M<Mvalue>-N<Nvalue>`.
    Returns:

    """
    M, N, augment_operations = get_M_and_N(augment_type)
    selected = np.random.choice(augment_operations, N)

    for augmentation in selected:
        if augmentation == "none":
            continue

        if augmentation == "scale":
            Mmax = step_val(M, *scale_range_Mmax)
            Mmin = step_val_rev(M, *scale_range_Mmin)
            X_batch = random_scale(X_batch, Mmax, Mmin)

        if augmentation == "flip":
            Mval = step_val(M, *flip_range_M)
            X_batch = random_flip(X_batch, Mval)

        if augmentation == "drop":
            Mval = step_val(M, *drop_range_M)
            X_batch = random_drop(X_batch, Mval)

        if augmentation == "cutout":
            Mval = step_val(M, *cutout_range_M)
            X_batch = random_cutout(X_batch, Mval)

        if augmentation == "shift":
            Mval = step_val(M, *shift_range_M)
            X_batch = random_shift(X_batch, Mval)

        if augmentation == "sine":
            Mval = step_val(M, *sine_noise_range_M)
            Fval = rand_val(*sine_noise_range_F)
            X_batch = sine_noise(X_batch, Mval, Fval)

        if augmentation == "square":
            Mval = step_val(M, *square_noise_range_M)
            Fval = rand_val(*square_noise_range_F)
            X_batch = square_noise(X_batch, Mval, Fval)

        if augmentation == "wn":
            Mval = step_val(M, *white_noise_range_M)
            X_batch = white_noise(X_batch, Mval)

        if augmentation == "sine_p":
            Mval = step_val(M, *partial_sine_noise_range_M)
            Fval = rand_val(*partial_sine_noise_range_F)
            X_batch = sine_noise_partial(X_batch, Mval, Fval)

        if augmentation == "square_p":
            Mval = step_val(M, *partial_square_noise_range_M)
            Fval = rand_val(*partial_square_noise_range_F)
            X_batch = square_noise_partial(X_batch, Mval, Fval)

        if augmentation == "wn_p":
            Mval = step_val(M, *partial_white_noise_range_M)
            X_batch = white_noise_partial(X_batch, Mval)

        if augmentation == "fir_l":
            Mmax = fir_low_range_M[1]
            Mmin = step_val_rev(M, *fir_low_range_M)
            X_batch = apply_FIR_low(X_batch, Mmax, Mmin)

    X_batch = X_batch.astype(np.float32)
    return X_batch


def fix_aug(X_batch, augmentation, M):
    """
    Apply fixed augmentaion function.
    """

    if augmentation == "scale":
        Mmax = step_val(M, *scale_range_Mmax)
        Mmin = step_val_rev(M, *scale_range_Mmin)
        X_batch = random_scale(X_batch, Mmax, Mmin)

    if augmentation == "flip":
        Mval = step_val(M, *flip_range_M)
        X_batch = random_flip(X_batch, Mval)

    if augmentation == "drop":
        Mval = step_val(M, *drop_range_M)
        X_batch = random_drop(X_batch, Mval)

    if augmentation == "cutout":
        Mval = step_val(M, *cutout_range_M)
        X_batch = random_cutout(X_batch, Mval)

    if augmentation == "shift":
        Mval = step_val(M, *shift_range_M)
        X_batch = random_shift(X_batch, Mval)

    if augmentation == "sine":
        Mval = step_val(M, *sine_noise_range_M)
        Fval = rand_val(*sine_noise_range_F)
        X_batch = sine_noise(X_batch, Mval, Fval)

    if augmentation == "square":
        Mval = step_val(M, *square_noise_range_M)
        Fval = rand_val(*square_noise_range_F)
        X_batch = square_noise(X_batch, Mval, Fval)

    if augmentation == "wn":
        Mval = step_val(M, *white_noise_range_M)
        X_batch = white_noise(X_batch, Mval)

    if augmentation == "sine_p":
        Mval = step_val(M, *partial_sine_noise_range_M)
        Fval = rand_val(*partial_sine_noise_range_F)
        X_batch = sine_noise_partial(X_batch, Mval, Fval)

    if augmentation == "square_p":
        Mval = step_val(M, *partial_square_noise_range_M)
        Fval = rand_val(*partial_square_noise_range_F)
        X_batch = square_noise_partial(X_batch, Mval, Fval)

    if augmentation == "wn_p":
        Mval = step_val(M, *partial_white_noise_range_M)
        X_batch = white_noise_partial(X_batch, Mval)

    if augmentation == "fir_l":
        Mmax = fir_low_range_M[1]
        Mmin = step_val_rev(M, *fir_low_range_M)
        X_batch = apply_FIR_low(X_batch, Mmax, Mmin)

    X_batch = X_batch.astype(np.float32)
    return X_batch


def all_aug(X_batch, M):
    """
    Apply all augmentaion function.
    """

    Mmax = step_val(M, *scale_range_Mmax)
    Mmin = step_val_rev(M, *scale_range_Mmin)
    X_batch = random_scale(X_batch, Mmax, Mmin)

    Mval = step_val(M, *flip_range_M)
    X_batch = random_flip(X_batch, Mval)

    Mval = step_val(M, *drop_range_M)
    X_batch = random_drop(X_batch, Mval)

    Mval = step_val(M, *cutout_range_M)
    X_batch = random_cutout(X_batch, Mval)

    Mval = step_val(M, *shift_range_M)
    X_batch = random_shift(X_batch, Mval)

    Mval = step_val(M, *sine_noise_range_M)
    Fval = rand_val(*sine_noise_range_F)
    X_batch = sine_noise(X_batch, Mval, Fval)

    Mval = step_val(M, *square_noise_range_M)
    Fval = rand_val(*square_noise_range_F)
    X_batch = square_noise(X_batch, Mval, Fval)

    Mval = step_val(M, *white_noise_range_M)
    X_batch = white_noise(X_batch, Mval)

    Mval = step_val(M, *partial_sine_noise_range_M)
    Fval = rand_val(*partial_sine_noise_range_F)
    X_batch = sine_noise_partial(X_batch, Mval, Fval)

    Mval = step_val(M, *partial_square_noise_range_M)
    Fval = rand_val(*partial_square_noise_range_F)
    X_batch = square_noise_partial(X_batch, Mval, Fval)

    Mval = step_val(M, *partial_white_noise_range_M)
    X_batch = white_noise_partial(X_batch, Mval)

    Mmax = fir_low_range_M[1]
    Mmin = step_val_rev(M, *fir_low_range_M)
    X_batch = apply_FIR_low(X_batch, Mmax, Mmin)

    X_batch = X_batch.astype(np.float32)
    return X_batch


def exclude_aug(X_batch, augmentation, M):
    """
    Apply fixed augmentaion function.
    """

    if augmentation != "scale":
        Mmax = step_val(M, *scale_range_Mmax)
        Mmin = step_val_rev(M, *scale_range_Mmin)
        X_batch = random_scale(X_batch, Mmax, Mmin)

    if augmentation != "flip":
        Mval = step_val(M, *flip_range_M)
        X_batch = random_flip(X_batch, Mval)

    if augmentation != "drop":
        Mval = step_val(M, *drop_range_M)
        X_batch = random_drop(X_batch, Mval)

    if augmentation != "cutout":
        Mval = step_val(M, *cutout_range_M)
        X_batch = random_cutout(X_batch, Mval)

    if augmentation != "shift":
        Mval = step_val(M, *shift_range_M)
        X_batch = random_shift(X_batch, Mval)

    if augmentation != "sine":
        Mval = step_val(M, *sine_noise_range_M)
        Fval = rand_val(*sine_noise_range_F)
        X_batch = sine_noise(X_batch, Mval, Fval)

    if augmentation != "square":
        Mval = step_val(M, *square_noise_range_M)
        Fval = rand_val(*square_noise_range_F)
        X_batch = square_noise(X_batch, Mval, Fval)

    if augmentation != "wn":
        Mval = step_val(M, *white_noise_range_M)
        X_batch = white_noise(X_batch, Mval)

    if augmentation != "sine_p":
        Mval = step_val(M, *partial_sine_noise_range_M)
        Fval = rand_val(*partial_sine_noise_range_F)
        X_batch = sine_noise_partial(X_batch, Mval, Fval)

    if augmentation != "square_p":
        Mval = step_val(M, *partial_square_noise_range_M)
        Fval = rand_val(*partial_square_noise_range_F)
        X_batch = square_noise_partial(X_batch, Mval, Fval)

    if augmentation != "wn_p":
        Mval = step_val(M, *partial_white_noise_range_M)
        X_batch = white_noise_partial(X_batch, Mval)

    if augmentation != "fir_l":
        Mmax = fir_low_range_M[1]
        Mmin = step_val_rev(M, *fir_low_range_M)
        X_batch = apply_FIR_low(X_batch, Mmax, Mmin)

    X_batch = X_batch.astype(np.float32)
    return X_batch
