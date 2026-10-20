"""Per-image segmentation metrics beyond Dice: IoU, precision, recall, HD95, Boundary IoU, miss.

Definitions (also written to results/extended/METRICS.md). All masks are boolean arrays of the same shape
(256 x 256 in this project) and distances are in pixels.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import ndimage

FOUR_CONNECTED = ndimage.generate_binary_structure(2, 1)


def hd95_empty_value(size: int) -> float:
    """HD95 assigned to an empty prediction: the image diagonal, size * sqrt(2) (362.04 for 256)."""
    return size * math.sqrt(2.0)


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """True positives, false positives and false negatives (pixels)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    return {"tp": tp, "fp": int(pred.sum()) - tp, "fn": int(gt.sum()) - tp}


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    """2TP / (2TP + FP + FN); both masks empty -> 1."""
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2.0 * tp / denom


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    """TP / (TP + FP + FN); both masks empty -> 1 (consistent with Dice, so iou = dice / (2 - dice))."""
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def precision_from_counts(tp: int, fp: int) -> float:
    """TP / (TP + FP); 0 if nothing is predicted."""
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def recall_from_counts(tp: int, fn: int) -> float:
    """TP / (TP + FN); 0 if the ground truth is empty (does not occur in these datasets)."""
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def boundary(mask: np.ndarray) -> np.ndarray:
    """Boundary of a mask: M AND NOT erosion(M), 4-connectivity, border_value=0 (image borders count as boundary)."""
    m = mask.astype(bool)
    return m & ~ndimage.binary_erosion(m, structure=FOUR_CONNECTED, border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray, size: int | None = None) -> float:
    """95th-percentile Hausdorff distance between the boundaries of ``pred`` and ``gt`` (pixels).

    max(P95 of distances boundary(pred) -> boundary(gt), P95 of distances boundary(gt) -> boundary(pred)),
    distances from scipy.ndimage.distance_transform_edt, np.percentile with default interpolation.
    Empty prediction -> size * sqrt(2) (362.04 for 256). Both empty -> 0. Empty ground truth with a
    non-empty prediction is undefined and gets the same penalty value (does not occur in these datasets).
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    size = size or max(pred.shape)
    if not pred.any():
        return 0.0 if not gt.any() else hd95_empty_value(size)
    if not gt.any():
        return hd95_empty_value(size)
    bp, bg = boundary(pred), boundary(gt)
    to_gt = ndimage.distance_transform_edt(~bg)[bp]
    to_pred = ndimage.distance_transform_edt(~bp)[bg]
    return float(max(np.percentile(to_gt, 95), np.percentile(to_pred, 95)))


def boundary_band(mask: np.ndarray, d: float) -> np.ndarray:
    """Pixels of the mask whose distance to the nearest background pixel is <= d (mask padded with a 1-pixel zero border)."""
    m = mask.astype(bool)
    dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
    return m & (dist <= d)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, d: float = 5.0) -> float:
    """|B_d(pred) AND B_d(gt)| / |B_d(pred) OR B_d(gt)|; empty prediction -> 0 (also if both are empty)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any():
        return 0.0
    bp, bg = boundary_band(pred, d), boundary_band(gt, d)
    union = int(np.logical_or(bp, bg).sum())
    return 0.0 if union == 0 else float(np.logical_and(bp, bg).sum() / union)


def all_metrics(pred: np.ndarray, gt: np.ndarray, boundary_d: float = 5.0) -> Dict[str, float]:
    """Every per-image metric for one prediction/ground-truth pair."""
    c = confusion_counts(pred, gt)
    return {**c,
            "dice": dice_from_counts(**c), "iou": iou_from_counts(**c),
            "precision": precision_from_counts(c["tp"], c["fp"]), "recall": recall_from_counts(c["tp"], c["fn"]),
            "hd95": hd95(pred, gt), "boundary_iou": boundary_iou(pred, gt, boundary_d),
            "miss": int(not pred.astype(bool).any())}
