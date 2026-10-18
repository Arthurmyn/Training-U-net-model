"""Tests for the preprocessing steps of the methodology."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from audit import load, preprocess
from audit.config import DatasetConfig

from .conftest import H, W, make_dataset, mask_array, textured


def test_image_is_resized_bilinear_and_scaled_to_unit_range():
    img = preprocess.preprocess_image(textured(1, 100, 60), 32)
    assert img.shape == (32, 32, 3) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    white = preprocess.preprocess_image(Image.new("RGB", (50, 20), (255, 255, 255)), 16)
    assert np.allclose(white, 1.0)
    black = preprocess.preprocess_image(Image.new("RGB", (50, 20), (0, 0, 0)), 16)
    assert np.allclose(black, 0.0)


def test_mask_uses_nearest_neighbour_and_threshold():
    raw = mask_array([(10, 8, 30, 24)])
    raw[0, 0] = 120                                    # noisy value below the threshold
    raw[1, 1] = 130                                    # noisy value above the threshold
    m = preprocess.preprocess_mask(raw, 24, 127)
    assert m.shape == (24, 24) and set(np.unique(m)) <= {0, 1} and m.dtype == np.uint8
    # nearest neighbour: no grey (interpolated) values can appear, area stays close
    assert abs(m.mean() - (raw > 127).mean()) < 0.03


def test_nearest_vs_bilinear_difference_on_masks():
    raw = np.zeros((40, 40), np.uint8)
    raw[::2, ::2] = 255                                # fine checker pattern: bilinear would blur to grey
    m = preprocess.preprocess_mask(raw, 10, 127)
    assert set(np.unique(m)) <= {0, 1}


def test_check_preprocessing_summary_and_examples(tmp_path):
    root = make_dataset(tmp_path / "d", n_good=12, defects=False)
    ds = DatasetConfig(key="s", name="S", root=root, image_dir="images", mask_dir="masks", expected_count=12,
                       split_method="random")
    files = load.discover(ds)
    df, s = preprocess.check_preprocessing(files, files.matched, size=32, threshold=127)
    assert s["ok"] and s["violations"] == [] and s["n_checked"] == 12
    assert s["image_min"] >= 0 and s["image_max"] <= 1 and s["mask_values"] == [0, 1]
    assert s["aspect_ratio"]["median"] == pytest.approx(W / H) and s["n_resized_non_square"] == 12
    ex = preprocess.select_examples(df, 5)
    assert len(ex) == 5 and ex == preprocess.select_examples(df, 5)
    areas = df.set_index("stem").loc[ex, "area_before"].tolist()
    assert areas == sorted(areas) and areas[0] == df["area_before"].min() and areas[-1] == df["area_before"].max()


def test_tiny_polyp_vanishing_is_a_violation(tmp_path):
    root = make_dataset(tmp_path / "d", n_good=2, defects=False)
    Image.fromarray(mask_array([(5, 5, 6, 6)])).save(root / "masks" / "g00.png")     # 1 pixel polyp
    ds = DatasetConfig(key="s", name="S", root=root, image_dir="images", mask_dir="masks", expected_count=2, split_method="random")
    files = load.discover(ds)
    _, s = preprocess.check_preprocessing(files, files.matched, size=8, threshold=127)
    assert not s["ok"] and s["masks_empty_after"] == ["g00"]
