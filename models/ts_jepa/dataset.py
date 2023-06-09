import torch
from torchvision import datasets, transforms
import logging
import os
import random
import time
import numpy as np
from functools import partial
import math
from typing import Tuple

from shutil import copyfile

logger = logging.getLogger(__name__)


def compute_block_mask_2d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:
    assert mask_length > 1

    B, L = shape

    d = int(L**0.5)

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(d / mask_length)
        inp_len = sz * sz

        inp = torch.zeros((B, 1, sz, sz))
        w = torch.ones((1, 1, mask_length, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > d:
            mask = mask[..., :d, :d]
    else:
        mask = torch.zeros((B, d, d))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length**2)
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d - 1)

        mask[(i0, i1, i2)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.zeros((1, 1, 3, 3))
        w[..., 0, 1] = 1
        w[..., 2, 1] = 1
        w[..., 1, 0] = 1
        w[..., 1, 2] = 1

        all_nbs = get_nbs(B, mask, w)

    mask = mask.reshape(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.view(1, d, d), w).flatten()

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


def compute_block_mask_1d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:
    B, L = shape

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(L / mask_length)

        inp = torch.zeros((B, 1, sz))
        w = torch.ones((1, 1, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(sz * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose1d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > L:
            mask = mask[..., :L]

    else:
        mask = torch.zeros((B, L))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length)
                    * (1 + mask_dropout)
                ),
            ),
        )

        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [])

        offset = mask_length // 2
        for i in range(mask_length):
            k1 = i - offset
            inds[0].append(centers[0])
            inds[1].append(centers[1] + k1)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=L - 1)

        mask[(i0, i1)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv1d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.ones((1, 1, 3))
        w[..., 1] = 0
        all_nbs = get_nbs(B, mask, w)

    mask = mask.view(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.unsqueeze(0), w).squeeze(0)

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


def load(path, loader, cache):
    if hasattr(caching_loader, "cache_root"):
        cache = caching_loader.cache_root

    cached_path = cache + path

    num_tries = 3
    for curr_try in range(num_tries):
        try:
            if curr_try == 2:
                return loader(path)
            if not os.path.exists(cached_path) or curr_try > 0:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                copyfile(path, cached_path)
                os.chmod(cached_path, 0o777)
            return loader(cached_path)
        except Exception as e:
            logger.warning(str(e))
            if "Errno 13" in str(e):
                caching_loader.cache_root = f"/scratch/{random.randint(0, 69420)}"
                logger.warning(f"setting cache root to {caching_loader.cache_root}")
                cached_path = caching_loader.cache_root + path
            if curr_try == (num_tries - 1):
                raise
            time.sleep(2)


def caching_loader(cache_root: str, loader):
    if cache_root is None:
        return loader

    if cache_root == "slurm_tmpdir":
        cache_root = os.environ["SLURM_TMPDIR"]
        assert len(cache_root) > 0

    if not cache_root.endswith("/"):
        cache_root += "/"

    return partial(load, loader=loader, cache=cache_root)


class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        second_size=None,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        second_interpolation="lanczos",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            logger.warning("range should be of kind (min, max)")

        if interpolation == "random":
            from PIL import Image

            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self._pil_interp(interpolation)

        self.second_interpolation = (
            self._pil_interp(second_interpolation)
            if second_interpolation is not None
            else None
        )
        self.scale = scale
        self.ratio = ratio

    def _pil_interp(self, method):
        from PIL import Image

        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return Image.BILINEAR

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        import torchvision.transforms.functional as F

        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(
                img, i, j, h, w, self.size, interpolation
            ), F.resized_crop(
                img, i, j, h, w, self.second_size, self.second_interpolation
            )


class MaeImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        local_cache_path=None,
        shuffle=True,
        key="imgs",
        beit_transforms=False,
        target_transform=False,
        no_transform=False,
        compute_mask=False,
        patch_size: int = 16,
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        dataset_type: str = "imagefolder",
    ):
        super().__init__()
        self.shuffle = shuffle
        self.key = key

        loader = caching_loader(local_cache_path, datasets.folder.default_loader)

        self.transform_source = None
        self.transform_target = None

        if target_transform:
            self.transform_source = transforms.ColorJitter(0.4, 0.4, 0.4)
            self.transform_target = transforms.ColorJitter(0.4, 0.4, 0.4)

        if no_transform:
            if input_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(input_size / crop_pct)

            self.transform_train = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=3),
                    transforms.CenterCrop(input_size),
                ]
            )

            self.transform_train = transforms.Resize((input_size, input_size))
        elif beit_transforms:
            beit_transform_list = []
            if not target_transform:
                beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
            beit_transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandomResizedCropAndInterpolationWithTwoPic(
                        size=input_size,
                        second_size=None,
                        interpolation="bicubic",
                        second_interpolation=None,
                    ),
                ]
            )
            self.transform_train = transforms.Compose(beit_transform_list)
        else:
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.2, 1.0), interpolation=3
                    ),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                ]
            )
        self.final_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if dataset_type == "imagefolder":
            self.dataset = datasets.ImageFolder(
                os.path.join(root, split), loader=loader
            )
        elif dataset_type == "path":
            self.dataset = PathDataset(
                root,
                loader,
                None,
                None,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            raise Exception(f"invalid dataset type {dataset_type}")

        logger.info(
            f"initial transform: {self.transform_train}, "
            f"source transform: {self.transform_source}, "
            f"target transform: {self.transform_target}, "
            f"final transform: {self.final_transform}"
        )
        logger.info(f"loaded {len(self.dataset)} examples")

        self.is_compute_mask = compute_mask
        self.patches = (input_size // patch_size) ** 2
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        img = self.transform_train(img)

        source = None
        target = None
        if self.transform_source is not None:
            source = self.final_transform(self.transform_source(img))
        if self.transform_target is not None:
            target = self.final_transform(self.transform_target(img))

        if source is None:
            img = self.final_transform(img)

        v = {"id": index, self.key: source if source is not None else img}
        if target is not None:
            v["target"] = target

        if self.is_compute_mask:
            if self.mask_length == 1:
                mask = compute_block_mask_1d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                )
            else:
                mask = compute_block_mask_2d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                    expand_adjcent=self.expand_adjacent,
                    mask_dropout=self.mask_dropout,
                    non_overlapping=self.non_overlapping,
                )

            v["precomputed_mask"] = mask

        return v

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s[self.key] for s in samples], dim=0)

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                self.key: collated_img,
            },
        }

        if "target" in samples[0]:
            collated_target = torch.stack([s["target"] for s in samples], dim=0)
            res["net_input"]["target"] = collated_target

        if "precomputed_mask" in samples[0]:
            collated_mask = torch.cat([s["precomputed_mask"] for s in samples], dim=0)
            res["net_input"]["precomputed_mask"] = collated_mask

        return res

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    @property
    def sizes(self):
        return np.full((len(self),), 1)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]


class MaeTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        input_size,
        shuffle=True,
        key="imgs",
        compute_mask=False,
        patch_size: int = 16,
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.key = key

        self.dataset = dataset

        self.is_compute_mask = compute_mask
        self.patches = input_size // patch_size
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        source = None
        target = None

        v = {"id": index, self.key: source if source is not None else img}
        if target is not None:
            v["target"] = target

        if self.is_compute_mask:
            # if self.mask_length == 1:
            mask = compute_block_mask_1d(
                shape=(self.clone_batch * 7, self.patches),  # TODO use in channels here
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
            )

            v["precomputed_mask"] = mask

        return v

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s[self.key] for s in samples], dim=0)

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                self.key: collated_img,
            },
        }

        if "target" in samples[0]:
            collated_target = torch.stack([s["target"] for s in samples], dim=0)
            res["net_input"]["target"] = collated_target

        if "precomputed_mask" in samples[0]:
            collated_mask = torch.cat([s["precomputed_mask"] for s in samples], dim=0)
            res["net_input"]["precomputed_mask"] = collated_mask

        return res

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    @property
    def sizes(self):
        return np.full((len(self),), 1)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]
