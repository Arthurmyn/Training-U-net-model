"""Tests for discovery and per-image checks on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit import checks, load
from audit.config import AuditConfig, DatasetConfig

from .conftest import H, W, mask_array, textured


def _cfg(root) -> tuple:
    ds = DatasetConfig(key="syn", name="Synthetic", root=root, image_dir="images", mask_dir="masks",
                       expected_count=10, split_method="random", split_sizes={"train": 1, "val": 1, "test": 1})
    cfg = AuditConfig(seed=1, output_dir=root / "out", mask_binarize=127, full_mask_fraction=0.9,
                      near_duplicate_hamming=4, datasets={"syn": ds}, base_dir=root)
    return cfg, ds


@pytest.fixture()
def table(synthetic):
    cfg, ds = _cfg(synthetic)
    files = load.discover(ds)
    return checks.audit_dataset(cfg, ds, files, {}, progress=False).set_index("stem"), files, cfg, ds


def test_discovery_matches_by_stem(synthetic):
    _, ds = _cfg(synthetic)
    f = load.discover(ds)
    assert f.unmatched_images == ["only_image"]
    assert f.unmatched_masks == ["only_mask"]
    assert "noisy" in f.matched          # image .png matched with mask .jpg
    assert len(f.images) == len(f.matched) + 1 and len(f.masks) == len(f.matched) + 1


def test_natural_order():
    assert sorted(["10", "2", "1", "100"], key=load.natural_key) == ["1", "2", "10", "100"]


def test_corrupt_file_is_flagged_not_fatal(table):
    t, *_ = table
    assert not t.loc["corrupt", "image_ok"]
    assert "corrupt_image" in t.loc["corrupt", "flags"]
    assert "UnidentifiedImageError" in t.loc["corrupt", "read_error"] or t.loc["corrupt", "read_error"]
    assert t.loc["g00", "image_ok"]


def test_noisy_jpeg_mask_detected_and_binarised(table):
    t, *_ = table
    r = t.loc["noisy"]
    assert r["mask_noisy"] and r["mask_n_unique_values"] > 2
    assert r["mask_n_classes_binarized"] == 2
    assert not t.loc["g00", "mask_noisy"]
    assert 0.2 < r["polyp_area_fraction"] < 0.5          # 20x16 box of 48x32 ~ 0.21


def test_area_fraction_and_components_exact(table):
    t, *_ = table
    assert t.loc["g00", "polyp_area_fraction"] == pytest.approx(10 * 10 / (W * H))
    assert t.loc["g00", "n_polyps"] == 1
    assert t.loc["two", "n_polyps"] == 2


def test_empty_and_full_masks(table):
    t, *_ = table
    assert t.loc["empty", "mask_empty"] and "empty_mask" in t.loc["empty", "flags"]
    assert t.loc["empty", "mask_n_classes_binarized"] == 1
    assert t.loc["full", "mask_near_full"] and "near_full_mask" in t.loc["full", "flags"]
    assert not t.loc["g01", "mask_empty"] and not t.loc["g01", "mask_near_full"]


def test_size_mismatch(table):
    t, *_ = table
    assert t.loc["mismatch", "size_mismatch"] and "size_mismatch" in t.loc["mismatch", "flags"]
    assert not t.loc["g00", "size_mismatch"]


def test_brightness_and_channels(table):
    t, *_ = table
    img = np.asarray(textured(0).convert("L"), dtype=float)
    assert t.loc["g00", "brightness_gray"] == pytest.approx(img.mean())
    rgb = np.asarray(textured(0), dtype=float)
    assert t.loc["g00", "mean_r"] == pytest.approx(rgb[..., 0].mean())


def test_summary_counts(table):
    t, files, cfg, ds = table
    s = checks.summarize_dataset(t.reset_index(), files, ds)
    assert s["counts"]["unmatched_images"] == 1 and s["counts"]["unmatched_masks"] == 1
    assert s["counts"]["matches_expected"] is False
    assert s["unreadable"]["corrupt_images"] == ["corrupt.png"]
    assert s["mask_values"]["n_masks_other_values"] >= 1
    assert s["empty_masks"] == ["empty"]


def test_exact_and_near_duplicates(table):
    t, files, cfg, ds = table
    df = t.reset_index()
    res = checks.find_duplicates({"syn": df}, 4)
    assert any(set(g["files"]) == {"syn/g00.png", "syn/dup_of_g00.png"} for g in res["exact_groups"])
    pairs = res["near_pairs"]
    row = pairs[(pairs.a_file.isin(["g00.png", "dup_of_g00.png"])) & (pairs.b_file.isin(["g00.png", "dup_of_g00.png"]))]
    assert len(row) == 1 and row.iloc[0]["hamming"] == 0 and bool(row.iloc[0]["exact_bytes"])


def test_near_duplicate_threshold_and_cross_dataset(tmp_path):
    from PIL import Image
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "images").mkdir(parents=True)
        (d / "masks").mkdir(parents=True)
    base = textured(7)
    base.save(a / "images" / "x.png")
    # same picture re-encoded as JPEG in the other dataset -> different bytes, near-identical content
    base.save(b / "images" / "y.png")
    arr = np.asarray(base).astype(int)
    Image.fromarray(np.clip(arr + 2, 0, 255).astype(np.uint8)).save(b / "images" / "y.png")
    textured(8).save(b / "images" / "z.png")
    for d, names in ((a, ["x"]), (b, ["y", "z"])):
        for n in names:
            Image.fromarray(mask_array([(5, 5, 15, 15)])).save(d / "masks" / f"{n}.png")
    frames = {}
    for key, d in (("a", a), ("b", b)):
        cfg, ds = _cfg(d)
        f = load.discover(ds)
        frames[key] = checks.audit_dataset(cfg, ds, f, {}, progress=False)
    res = checks.find_duplicates(frames, 4)
    cross = res["near_pairs"][res["near_pairs"].scope == "across_datasets"]
    assert len(cross) == 1 and {cross.iloc[0].a_file, cross.iloc[0].b_file} == {"x.png", "y.png"}
    assert not cross.iloc[0]["exact_bytes"]


def test_phash_is_deterministic_and_discriminative():
    h1, h2, h3 = checks.phash(textured(1)), checks.phash(textured(1)), checks.phash(textured(2))
    assert h1 == h2 and len(h1) == 16
    d = int(np.sum(checks.hex_to_bits([h1])[0] != checks.hex_to_bits([h3])[0]))
    assert d > 4
