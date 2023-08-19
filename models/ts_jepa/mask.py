# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


class RandomMaskCollator(object):
    def __init__(
        self,
        ratio=0.5,
        input_size=1000,
        patch_size=16,
        channel_independence=False,
    ):
        super(RandomMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.ratio = ratio
        self.channel_independence = channel_independence

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """

        collated_batch = torch.utils.data.default_collate(batch)
        collated_X = collated_batch[0]
        collated_y = collated_batch[1]

        # TODO actually just for the forecasting datasets, otherwise collated y has to be handled differently
        bs, num_patches, feat_dim = collated_X.shape

        # TODO use CIDataset class wrapper instead of collate fn for channel independence
        # CI dataset should already make sure that there are only univariate ts at this point
        if self.channel_independence and feat_dim > 1:
            collated_X = (
                collated_X.transpose(1, 2)
                .reshape(bs * feat_dim, num_patches)
                .unsqueeze(-1)
            )
            collated_y = (
                collated_y.transpose(1, 2).reshape(bs * feat_dim, -1).unsqueeze(-1)
            )

        B = collated_X.shape[0]

        num_keep = int(self.num_patches * (1.0 - self.ratio))

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            m = torch.randperm(self.num_patches)
            collated_masks_enc.append([m[:num_keep]])
            collated_masks_pred.append([m[num_keep:]])

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_X, collated_y, collated_masks_enc, collated_masks_pred


class BlockMaskCollator(object):
    def __init__(
        self,
        input_size,
        patch_size,
        enc_mask_scale,
        pred_mask_scale,
        nenc=1,
        npred=1,
        # min_keep=4,
        allow_overlap=False,
        channel_independence=False,
    ):
        super(BlockMaskCollator, self).__init__()
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        # minimum number of patches to keep
        # TODO not hardcoded, e.g. relate to enc_mask_scale
        self.min_keep = round(0.1 * self.num_patches)
        # whether to allow overlap b/w enc and pred masks
        self.allow_overlap = allow_overlap
        self.channel_independence = channel_independence

    def _sample_block_size(self, scale):
        _rand = torch.rand(1).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.num_patches * mask_scale)

        return max_keep

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False

        while not valid_mask:
            # sample block from beginning
            # all patches zero except for the randomly selected block
            if self.num_patches == b_size:
                left = 0
            else:
                left = torch.randint(0, self.num_patches - b_size, (1,))

            mask = torch.zeros(self.num_patches, dtype=torch.int32)
            mask[left : left + b_size] = 1

            # constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)

            # return index for non-zero values (randomly selected block is 1, rest is 0)
            mask = torch.nonzero(mask.flatten())

            # if mask too small try again
            valid_mask = len(mask) >= self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(self.num_patches, dtype=torch.int32)
        mask_complement[left : left + b_size] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        # channel independence
        collated_batch = torch.utils.data.default_collate(batch)
        collated_X = collated_batch[0]
        collated_y = collated_batch[1]

        # TODO actually just for the forecasting datasets, otherwise collated y has to be handled differently
        if self.channel_independence:
            bs, num_patches, feat_dim = collated_X.shape
            collated_X = (
                collated_X.transpose(1, 2)
                .reshape(bs * feat_dim, num_patches)
                .unsqueeze(-1)
            )
            collated_y = (
                collated_y.transpose(1, 2).reshape(bs * feat_dim, -1).unsqueeze(-1)
            )

        B = collated_X.shape[0]

        p_size = self._sample_block_size(scale=self.pred_mask_scale)
        e_size = self._sample_block_size(scale=self.enc_mask_scale)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.num_patches
        min_keep_enc = self.num_patches

        for _ in range(B):
            # number of prediction masks for the same encoding
            masks, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks)

            # use complement masks as acceptable regions to enforce no overlap
            acceptable_regions = masks_C
            if self.allow_overlap:
                acceptable_regions = None

            # number of encoding masks
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_X, collated_y, collated_masks_enc, collated_masks_pred
