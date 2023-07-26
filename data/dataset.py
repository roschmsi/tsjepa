# References:
# https://github.com/gzerveas/mvts_transformer
# https://github.com/yuqinie98/PatchTST

import numpy as np
from torch.utils.data import Dataset
import torch

from data.ecg_dataset import load_ecg_dataset
from data.fc_dataset import load_fc_dataset


def load_dataset(config):
    if config.dataset in ["ecg", "ptb-xl", "ecg5"]:
        return load_ecg_dataset(config)
    elif config.dataset in [
        "etth1",
        "etth2",
        "ettm1",
        "ettm2",
        "illness",
        "traffic",
        "weather",
        "electricity",
    ]:
        return load_fc_dataset(config)
    else:
        raise ValueError("Dataset type is not specified")


class PretrainingDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(
        self,
        ecg_dataset,
        mean_mask_length,
        masking_ratio,
        mode="separate",
        distribution="geometric",
        exclude_feats=None,
    ):
        super(PretrainingDataset, self).__init__()
        self.ecg_dataset = ecg_dataset
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
            ID: ID of sample
        """
        X, _ = self.ecg_dataset.__getitem__(ind)

        mask = noise_mask(
            X=X,
            masking_ratio=self.masking_ratio,
            mean_mask_length=self.mean_mask_length,
            mode=self.mode,
            distribution=self.distribution,
            exclude_feats=self.exclude_feats,
        )  # (seq_length, feat_dim) boolean array

        return torch.from_numpy(X), torch.from_numpy(mask)

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.ecg_dataset)


class ClassificationDataset(Dataset):
    def __init__(self, ecg_dataset):
        super(ClassificationDataset, self).__init__()
        self.ecg_dataset = ecg_dataset

    def __getitem__(self, ind):
        X, y = self.ecg_dataset.__getitem__(ind)
        return torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.ecg_dataset)


class ForecastingDataset(Dataset):
    def __init__(self, dataset):
        super(ForecastingDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, ind):
        X, y = self.dataset.__getitem__(ind)
        return X, y

    def __len__(self):
        return len(self.dataset)


# class JEPADataset(Dataset):
#     def __init__(self, dataset, num_channels=12):
#         super(JEPADataset, self).__init__()
#         self.dataset = dataset
#         self.num_channels = num_channels

#     def __getitem__(self, ind):
#         n = ind // self.num_channels
#         c = ind % self.num_channels
#         X, _ = self.dataset.__getitem__(n)
#         X = X[:, c]
#         X = np.expand_dims(X, 1).astype(float)
#         return X

#     def __len__(self):
#         return len(self.dataset) * self.num_channels


class JEPADataset(Dataset):
    def __init__(self, dataset, num_channels=12):
        super(JEPADataset, self).__init__()
        self.dataset = dataset
        self.num_channels = num_channels

    def __getitem__(self, ind):
        X, y = self.dataset.__getitem__(ind)
        X = X.astype(float)
        return X, y

    def __len__(self):
        return len(self.dataset)


class ClassificationPatchDataset(Dataset):
    def __init__(self, dataset, patch_len=16, stride=8):
        super(ClassificationPatchDataset, self).__init__()
        self.dataset = dataset
        self.patch_len = patch_len
        self.stride = stride

    def __getitem__(self, ind):
        X, y = self.dataset.__getitem__(ind)
        X = torch.from_numpy(X).unsqueeze(0)
        X = create_patch(X, self.patch_len, self.stride)
        return X.squeeze(), torch.from_numpy(y)

    def __len__(self):
        return len(self.dataset)


class ForecastingPatchDataset(Dataset):
    def __init__(self, dataset, patch_len=16, stride=8):
        super(ForecastingPatchDataset, self).__init__()
        self.dataset = dataset
        self.patch_len = patch_len
        self.stride = stride

    def __getitem__(self, ind):
        X, y = self.dataset.__getitem__(ind)
        # X = torch.from_numpy(X).unsqueeze(0)
        X = X.unsqueeze(0)
        X = create_patch(X, self.patch_len, self.stride)
        return X.squeeze(), y  # torch.from_numpy(y)

    def __len__(self):
        return len(self.dataset)


def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    # padding mask was dtype=torch.int16 and thus negative when over 32000
    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int32), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets.float(), padding_masks


def collate_patch_superv(
    data, max_len=None, patch_len=None, stride=None, masking_ratio=0
):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    num_patch = (max(max_len, patch_len) - patch_len) // stride + 1
    num_patch = int((1 - masking_ratio) * num_patch)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    max_len = num_patch
    X = torch.zeros(
        batch_size, max_len, features[0].shape[1], features[0].shape[2]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        # draw random index, not ecg data from beginning
        X[i, :end, :, :] = features[i][:end, :, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    # padding mask was dtype=torch.int16 and thus negative when over 32000
    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int32), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets.float(), padding_masks


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(
        num_active, torch.ones(num_active.shape, dtype=torch.int16)
    )  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def collate_unsuperv(data, max_len=None, mask_compensation=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(
        X, dtype=torch.bool
    )  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int32), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks


def collate_patch_unsuperv(
    data, feat_dim, max_len, patch_len, stride, masking_ratio=0, num_levels=None
):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, X_kept, targets, masks, ids_restore = zip(*data)

    num_patch = (max(max_len, patch_len) - patch_len) // stride + 1

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    max_len = num_patch

    lengths_kept = [X.shape[0] for X in X_kept]
    max_len_kept = int(num_patch * (1 - masking_ratio))

    X_col = torch.zeros(
        batch_size, max_len, features[0].shape[1], features[0].shape[2]
    )  # (batch_size, padded_length, feat_dim)
    X_kept_padded = torch.zeros(
        batch_size, max_len_kept, X_kept[0].shape[1], X_kept[0].shape[2]
    )

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X_col[i, :end, :, :] = features[i][:end, :, :]

        end_kept = min(lengths_kept[i], max_len_kept)
        X_kept_padded[i, :end_kept, :, :] = X_kept[i][:end_kept, :, :]

    # features = torch.stack(features)
    targets_col = torch.zeros(
        batch_size, max_len, features[0].shape[1], features[0].shape[2]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        targets_col[i, :end, :, :] = targets[i][:end, :, :]

    masks_col = torch.zeros(batch_size, max_len, features[0].shape[1])
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        masks_col[i, :end, :] = masks[i][:end, :]

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int32), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep
    # target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict

    padding_masks_kept = padding_mask(
        torch.tensor(lengths_kept, dtype=torch.int32), max_len=max_len_kept
    )

    ids_restore_max_len = (
        max_len if num_levels is None else max_len // (2 ** (num_levels - 1))
    )
    ids_restore = torch.stack(
        [
            torch.cat(
                [
                    ids,
                    torch.zeros(
                        ids_restore_max_len - len(ids), feat_dim, dtype=torch.int
                    ),
                ]
            )
            for ids in ids_restore
        ]
    )

    return (
        X_col,
        X_kept_padded,
        targets_col,
        masks_col,
        padding_masks,
        padding_masks_kept,
        ids_restore,
    )


def noise_mask(
    X,
    masking_ratio,
    mean_mask_length,
    mode="separate",
    distribution="geometric",
    exclude_feats=None,
):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == "geometric":  # stateful (Markov chain)
        if mode == "separate":  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(
                        X.shape[0], mean_mask_length, masking_ratio
                    )  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(
                np.expand_dims(
                    geom_noise_mask_single(X.shape[0], mean_mask_length, masking_ratio),
                    1,
                ),
                X.shape[1],
            )
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == "separate":
            mask = np.random.choice(
                np.array([True, False]),
                size=X.shape,
                replace=True,
                p=(1 - masking_ratio, masking_ratio),
            )
        else:
            mask = np.tile(
                np.random.choice(
                    np.array([True, False]),
                    size=(X.shape[0], 1),
                    replace=True,
                    p=(1 - masking_ratio, masking_ratio),
                ),
                X.shape[1],
            )

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)

    # probability of each masking sequence stopping. parameter of geometric distribution.
    p_m = 1 / lm
    # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)

    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    # state 0 means masking, 1 means not masking
    state = int(np.random.rand() > masking_ratio)

    for i in range(L):
        keep_mask[
            i
        ] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


class PretrainingPatchDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(
        self,
        dataset,
        masking_ratio,
        patch_len,
        stride,
        debug=False,
        only_time_masking=False,
        hierarchical=False,
        num_levels=None,
    ):
        super(PretrainingPatchDataset, self).__init__()
        self.dataset = dataset
        self.masking_ratio = masking_ratio
        self.patch_len = patch_len
        self.stride = stride
        self.debug = debug
        self.only_time_masking = only_time_masking
        self.hierarchical = hierarchical
        self.num_levels = num_levels

    def __getitem__(self, idx):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
            ID: ID of sample
        """
        X, _ = self.dataset.__getitem__(idx)
        if type(X) == torch.Tensor:
            X = X.unsqueeze(0)
        else:
            X = torch.from_numpy(X).unsqueeze(0)

        bs, seq_len, feat_dim = X.shape

        # ensure that consecutive patches are masked for hierarchical pretraining
        patch_len = self.patch_len
        stride = self.stride
        if self.hierarchical:
            patch_len = self.patch_len * (2 ** (self.num_levels - 1))
            stride = patch_len

        X = create_patch(X, patch_len=patch_len, stride=stride)
        if self.only_time_masking:
            X_masked, X_kept, mask, ids_restore = random_patch_masking_only_time(
                X, self.masking_ratio, debug=self.debug
            )
        else:
            X_masked, X_kept, mask, ids_restore = random_patch_masking(
                X, self.masking_ratio, debug=self.debug
            )

        # reshape
        if self.hierarchical:
            X_masked = (
                X_masked.transpose(2, 3)
                .reshape(bs, -1, feat_dim)
                .reshape(bs, -1, self.patch_len, feat_dim)
                .transpose(2, 3)
            )

            X_kept = (
                X_kept.transpose(2, 3)
                .reshape(bs, -1, feat_dim)
                .reshape(bs, -1, self.patch_len, feat_dim)
                .transpose(2, 3)
            )

            X = (
                X.transpose(2, 3)
                .reshape(bs, -1, feat_dim)
                .reshape(bs, -1, self.patch_len, feat_dim)
                .transpose(2, 3)
            )

            mask = mask.repeat_interleave(repeats=2 ** (self.num_levels - 1), dim=1)

        return (
            X_masked.squeeze(),
            X_kept.squeeze(),
            X.squeeze(),
            mask.squeeze(),
            ids_restore.squeeze(),
        )

    def __len__(self):
        return len(self.dataset)


def random_patch_masking(xb, mask_ratio, debug):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    if debug:
        noise = torch.rand(
            size=(bs, L, nvars), device=xb.device, generator=torch.Generator()
        )  # noise in [0, 1], bs x L x nvars
    else:
        noise = torch.rand(
            size=(bs, L, nvars),
            device=xb.device,
        )

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D)
    )  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(
        bs, L - len_keep, nvars, D, device=xb.device
    )  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D)
    )  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def random_patch_masking_only_time(xb, mask_ratio, debug):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    if debug:
        noise = torch.rand(
            size=(bs, L, 1), device=xb.device, generator=torch.Generator()
        )  # noise in [0, 1], bs x L x nvars
    else:
        noise = torch.rand(
            size=(bs, L, 1),
            device=xb.device,
        )

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_shuffle = ids_shuffle.expand(-1, -1, nvars)
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D)
    )  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(
        bs, L - len_keep, nvars, D, device=xb.device
    )  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D)
    )  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    xb = xb.unfold(
        dimension=1, size=patch_len, step=stride
    )  # xb: [bs x num_patch x n_vars x patch_len]
    return xb
