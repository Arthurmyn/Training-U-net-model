"""The methodology's preprocessing (resize to size x size, bilinear for images, nearest for masks,
binarise masks at > threshold, scale pixels to [0, 1]) and a check that it behaves as intended.

Geometric augmentation is deliberately not part of this module: it is applied only during training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .load import Discovery, natural_key, open_image_rgb, read_mask_raw


def preprocess_image(rgb: Image.Image, size: int = 256) -> np.ndarray:
    """RGB image -> float32 array (size, size, 3) in [0, 1]; Pillow bilinear (antialiased when shrinking)."""
    return np.asarray(rgb.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def preprocess_mask(raw: np.ndarray, size: int = 256, threshold: int = 127) -> np.ndarray:
    """Raw mask values -> uint8 array (size, size) with values {0, 1}: nearest-neighbour resize, then value > threshold."""
    resized = Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return (np.asarray(resized) > threshold).astype(np.uint8)


def preprocess_pair(image_path: Path, mask_path: Path, size: int = 256, threshold: int = 127
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess one image/mask pair."""
    return preprocess_image(open_image_rgb(image_path), size), preprocess_mask(read_mask_raw(mask_path)[0], size, threshold)


def check_preprocessing(files: Discovery, stems: Sequence[str], size: int = 256, threshold: int = 127,
                        area_tolerance: float = 0.01) -> Tuple[pd.DataFrame, dict]:
    """Run the preprocessing on every given pair and measure what it changes.

    Returns ``(per_pair_table, summary)``. ``summary['violations']`` lists anything unexpected: wrong shape or
    dtype, pixel values outside [0, 1], mask values other than {0, 1}, masks that become empty, or a polyp area
    that changes by more than ``area_tolerance`` (absolute fraction of the image).
    """
    rows = []
    shape_bad = range_bad = 0
    for stem in stems:
        raw, _ = read_mask_raw(files.masks[stem])
        rgb = open_image_rgb(files.images[stem])
        img = preprocess_image(rgb, size)
        mask = preprocess_mask(raw, size, threshold)
        before = raw > threshold
        shape_bad += int(img.shape != (size, size, 3) or mask.shape != (size, size) or img.dtype != np.float32)
        range_bad += int(img.min() < 0.0 or img.max() > 1.0)
        rows.append({
            "stem": stem, "orig_width": rgb.width, "orig_height": rgb.height, "aspect_ratio": rgb.width / rgb.height,
            "area_before": float(before.mean()), "area_after": float(mask.mean()),
            "components_before": int(ndimage.label(before)[1]), "components_after": int(ndimage.label(mask)[1]),
            "image_min": float(img.min()), "image_max": float(img.max()),
            "mask_values": ";".join(str(int(v)) for v in np.unique(mask)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"size": size, "n_checked": 0, "violations": ["no pairs to check"]}
    df["area_abs_change"] = (df["area_after"] - df["area_before"]).abs()
    values = sorted({int(v) for s in df["mask_values"] for v in s.split(";")})
    empty_after = df.loc[(df["area_before"] > 0) & (df["area_after"] == 0), "stem"].tolist()
    summary = {
        "size": size, "mask_threshold": threshold, "n_checked": int(len(df)),
        "interpolation": {"images": "bilinear (Pillow)", "masks": "nearest"},
        "image_min": float(df["image_min"].min()), "image_max": float(df["image_max"].max()),
        "mask_values": values,
        "aspect_ratio": {"min": float(df["aspect_ratio"].min()), "median": float(df["aspect_ratio"].median()),
                         "max": float(df["aspect_ratio"].max())},
        "n_resized_non_square": int((df["aspect_ratio"].sub(1).abs() > 1e-6).sum()),
        "area_abs_change": {"mean": float(df["area_abs_change"].mean()), "max": float(df["area_abs_change"].max())},
        "masks_empty_after": empty_after,
        "masks_component_count_changed": int((df["components_before"] != df["components_after"]).sum()),
    }
    v: List[str] = []
    if shape_bad:
        v.append(f"{shape_bad} pair(s) with wrong output shape or dtype")
    if range_bad:
        v.append(f"{range_bad} image(s) with pixel values outside [0, 1]")
    if not set(values) <= {0, 1}:
        v.append(f"mask values after preprocessing are {values}, expected only 0 and 1")
    if empty_after:
        v.append(f"{len(empty_after)} mask(s) became empty after resizing")
    if summary["area_abs_change"]["max"] > area_tolerance:
        v.append(f"polyp area changed by up to {100 * summary['area_abs_change']['max']:.2f} percentage points (limit {100 * area_tolerance:.1f})")
    summary["violations"] = v
    summary["ok"] = not v
    return df, summary


def select_examples(df: pd.DataFrame, n: int = 5) -> List[str]:
    """Deterministic examples spread over the range of polyp sizes (smallest ... median ... largest)."""
    if df.empty:
        return []
    order = sorted(df.itertuples(), key=lambda r: (r.area_before, natural_key(r.stem)))
    idx = sorted({int(round(i)) for i in np.linspace(0, len(order) - 1, min(n, len(order)))})
    return [order[i].stem for i in idx]
