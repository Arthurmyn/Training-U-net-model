"""Tests for the extended metrics, the t-based gap statistics, the decision rule and Cohen's kappa."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from train import extended_metrics as em

SIZE = 256


def canvas(*pixels, shape=(32, 32)) -> np.ndarray:
    m = np.zeros(shape, bool)
    for r, c in pixels:
        m[r, c] = True
    return m


def test_counts_dice_iou_precision_recall_known_example():
    gt = canvas(*[(0, c) for c in range(5)])                     # 5 ground-truth pixels
    pred = canvas((0, 0), (0, 1), (0, 2), (4, 4))                # 3 hit, 1 wrong -> TP=3, FP=1, FN=2
    m = em.all_metrics(pred, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 2)
    assert m["dice"] == pytest.approx(0.6667, abs=1e-4)
    assert m["iou"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["miss"] == 0


def test_iou_equals_dice_over_two_minus_dice_on_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.random((16, 16)) > 0.6, rng.random((16, 16)) > 0.6
        m = em.all_metrics(a, b)
        assert m["iou"] == pytest.approx(m["dice"] / (2 - m["dice"]), abs=1e-9)


def test_both_empty_conventions():
    e = np.zeros((SIZE, SIZE), bool)
    m = em.all_metrics(e, e)
    assert m["dice"] == 1.0 and m["iou"] == 1.0 and m["hd95"] == 0.0 and m["miss"] == 1


def test_identical_masks_give_hd95_zero_and_boundary_iou_one():
    gt = np.zeros((SIZE, SIZE), bool)
    gt[60:140, 70:150] = True
    m = em.all_metrics(gt.copy(), gt)
    assert m["hd95"] == 0.0 and m["boundary_iou"] == pytest.approx(1.0) and m["dice"] == 1.0


def test_single_pixels_five_apart_give_hd95_five():
    assert em.hd95(canvas((10, 10)), canvas((10, 15))) == pytest.approx(5.0)
    assert em.hd95(canvas((10, 10)), canvas((15, 10))) == pytest.approx(5.0)


def test_empty_prediction_penalties():
    gt = np.zeros((SIZE, SIZE), bool)
    gt[100:120, 100:120] = True
    pred = np.zeros_like(gt)
    m = em.all_metrics(pred, gt)
    assert m["hd95"] == pytest.approx(256 * math.sqrt(2)) and round(m["hd95"], 2) == 362.04
    assert m["boundary_iou"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0 and m["miss"] == 1 and m["dice"] == 0.0


def test_boundary_uses_four_connectivity_and_counts_image_border():
    full = np.ones((8, 8), bool)
    b = em.boundary(full)
    assert b.sum() == 28 and not b[1:-1, 1:-1].any()             # only the outer ring (border_value=0)
    inner = np.zeros((9, 9), bool)
    inner[2:7, 2:7] = True
    assert em.boundary(inner).sum() == 16                        # 5x5 block: ring of 16
    diag = canvas((3, 3), (4, 4), shape=(9, 9))
    assert em.boundary(diag).sum() == 2                          # diagonal neighbours are not 4-connected: both are boundary


def test_narrow_masks_have_boundary_iou_equal_to_iou():
    gt = np.zeros((64, 64), bool)
    gt[10:13, 10:30] = True                                      # 3 pixels thick (narrower than 2d = 10)
    pred = np.zeros_like(gt)
    pred[11:14, 10:30] = True
    assert (em.boundary_band(gt, 5) == gt).all() and (em.boundary_band(pred, 5) == pred).all()
    assert em.boundary_iou(pred, gt, 5) == pytest.approx(em.iou_from_counts(**em.confusion_counts(pred, gt)))


def test_boundary_band_is_a_ring_for_large_masks():
    m = np.zeros((40, 40), bool)
    m[5:35, 5:35] = True                                         # 30x30 block
    band = em.boundary_band(m, 5)
    assert band.sum() == 30 * 30 - 20 * 20                      # distance <= 5 keeps a ring 5 pixels wide... inner 20x20 removed
    assert not band[15:25, 15:25].any()


def test_boundary_iou_is_less_forgiving_than_iou_for_large_shifted_masks():
    gt = np.zeros((128, 128), bool)
    gt[20:100, 20:100] = True
    pred = np.roll(gt, 6, axis=1)
    assert em.boundary_iou(pred, gt, 5) < em.iou_from_counts(**em.confusion_counts(pred, gt))
