"""Loss and the evaluation metric (Dice per image)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """1 - soft Dice per image, averaged over the batch."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy (on logits) + soft Dice loss."""
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


def dice_per_image(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Dice coefficient of each image: 2|P and G| / (|P| + |G|) after thresholding the probabilities.

    If both prediction and ground truth are empty the Dice is 1. If only one is empty it is 0.
    Returns a tensor of shape (B,).
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return torch.where(denom == 0, torch.ones_like(inter), 2.0 * inter / denom.clamp(min=1.0))
