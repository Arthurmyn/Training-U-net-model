"""Geometric augmentation for training only. Image and mask always get the same transform."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def augment(img: torch.Tensor, mask: torch.Tensor, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random flips plus a random rotation/zoom. ``img`` is (3, H, W) float, ``mask`` is (1, H, W) with values {0, 1}.

    Images use bilinear sampling, masks nearest neighbour (so masks stay binary). Uses torch's global RNG,
    which the DataLoader seeds per worker, so runs are reproducible for a given seed.
    """
    if torch.rand(1).item() < cfg.get("hflip", 0.0):
        img, mask = img.flip(-1), mask.flip(-1)
    if torch.rand(1).item() < cfg.get("vflip", 0.0):
        img, mask = img.flip(-2), mask.flip(-2)
    deg = float(cfg.get("rotation_deg", 0.0))
    smin, smax = cfg.get("scale", [1.0, 1.0])
    if (deg > 0 or smin != smax) and torch.rand(1).item() < cfg.get("p_affine", 0.0):
        angle = (torch.rand(1).item() * 2 - 1) * deg * math.pi / 180.0
        scale = smin + torch.rand(1).item() * (smax - smin)          # > 1 zooms in
        c, s = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=img.dtype).unsqueeze(0)
        grid = F.affine_grid(theta, (1,) + tuple(img.shape), align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        mask = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return img.contiguous(), mask.contiguous()
