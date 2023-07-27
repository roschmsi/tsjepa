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
    ):
        super(RandomMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.ratio = ratio

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)
        collated_X = collated_batch[0]
        collated_y = collated_batch[1]

        num_keep = int(self.num_patches * (1.0 - self.ratio))

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            m = torch.randperm(self.num_patches)
            collated_masks_enc.append([m[:num_keep]])
            collated_masks_pred.append([m[num_keep:]])

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_X, collated_y, collated_masks_enc, collated_masks_pred
