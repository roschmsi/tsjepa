"""Adapted from https://github.com/yuqinie98/PatchTST"""

import torch


def random_patch_masking(xb, mask_ratio, debug=False):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = round(L * (1 - mask_ratio))

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
    return x_masked, x_kept, mask, ids_keep, ids_restore


def block_patch_masking(xb, mask_ratio, debug=False):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = round(L * (1 - mask_ratio))

    # sort noise for each sample
    ids_shuffle = (
        torch.arange(L).repeat(bs, 1).unsqueeze(-1)
    )  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle  # ids_restore: [bs x L x nvars]

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
    return x_masked, x_kept, mask, ids_keep, ids_restore
