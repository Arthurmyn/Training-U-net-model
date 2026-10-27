"""Tests for the original-resolution robustness check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from train import original_resolution as orr  # noqa: E402
from train.config import load_train_config  # noqa: E402
from train.engine import train_run  # noqa: E402
from train.evaluate import evaluate_run  # noqa: E402

from .test_train import _project  # noqa: E402


def test_upsampling_a_probability_map_gives_a_mask_of_the_original_size():
    prob = torch.zeros(1, 4, 4)
    prob[:, :2, :2] = 0.9                                              # top-left quarter is polyp
    mask = orr.upsample_to_original(prob, 8, 12, 0.5)
    assert mask.shape == (8, 12) and mask.dtype == bool
    assert mask[:3, :5].all() and not mask[5:, 8:].any()               # quarter stays in its corner; edges are interpolated
    assert abs(mask.mean() - 0.25) < 0.1


def test_image_metrics_known_values():
    gt = np.zeros((10, 10), bool)
    gt[:5, :5] = True
    perfect = orr.image_metrics(gt.copy(), gt)
    assert perfect["dice"] == 1.0 and perfect["miss"] == 0
    empty = orr.image_metrics(np.zeros_like(gt), gt)
    assert empty["dice"] == 0.0 and empty["miss"] == 1 and empty["precision"] == 0.0


def test_run_on_tiny_project_compares_both_resolutions(tmp_path):
    cfg = load_train_config(_project(tmp_path)).override(**{"seeds": [1, 2], "training.max_epochs": 2})
    for source in ("a", "b"):
        for seed in (1, 2):
            evaluate_run(cfg, train_run(cfg, source, seed, progress=False), progress=False, examples=False)
    out = orr.run(cfg, progress=False)
    df = pd.read_csv(out / "per_image_metrics_original.csv")
    assert {"dice_original", "dice_256", "iou_original", "width", "height", "miss_original"} <= set(df.columns)
    assert df["dice_256"].notna().all() and df["dice_original"].between(0, 1).all()
    assert set(df["width"]) == {48} and set(df["height"]) == {32}      # original size of the synthetic images, not 256 x 256
    gap = pd.read_csv(out / "gap.csv")
    assert set(gap["resolution"]) == {"256 x 256", "original resolution"} and (gap["seeds"] == 2).all()
    assert (out / "README.md").is_file() and str(tmp_path) not in (out / "README.md").read_text(encoding="utf-8")
