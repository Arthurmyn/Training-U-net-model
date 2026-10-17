"""Synthetic datasets (tiny images) with deliberate defects for testing the audit."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

W, H = 48, 32


def textured(seed: int, w: int = W, h: int = H) -> Image.Image:
    """Smooth random RGB image; different seeds give clearly different perceptual hashes."""
    rng = np.random.default_rng(seed)
    low = rng.random((4, 4, 3))
    return Image.fromarray((low * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)


def mask_array(boxes, w: int = W, h: int = H) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = 255
    return m


def save_mask(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def make_dataset(root: Path, n_good: int = 10, defects: bool = True) -> Path:
    """Create ``root/images`` and ``root/masks`` with ``n_good`` clean pairs plus (optionally) defects.

    Defects: corrupt image, JPEG-noisy mask, empty mask, near-full mask, size mismatch,
    unmatched image, unmatched mask, exact duplicate image, two-polyp mask.
    """
    img_dir, msk_dir = root / "images", root / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    for i in range(n_good):
        textured(i).save(img_dir / f"g{i:02d}.png")
        save_mask(msk_dir / f"g{i:02d}.png", mask_array([(10, 8, 20 + i % 5, 18)]))
    if not defects:
        return root
    (img_dir / "corrupt.png").write_bytes(b"\x89PNG\r\n\x1a\nthis is not a real png")
    save_mask(msk_dir / "corrupt.png", mask_array([(5, 5, 15, 15)]))

    textured(101).save(img_dir / "noisy.png")                       # PNG image, JPEG mask
    buf = io.BytesIO()
    Image.fromarray(mask_array([(10, 8, 30, 24)])).save(buf, format="JPEG", quality=30)
    (msk_dir / "noisy.jpg").write_bytes(buf.getvalue())

    textured(102).save(img_dir / "empty.png")
    save_mask(msk_dir / "empty.png", mask_array([]))

    textured(103).save(img_dir / "full.png")
    save_mask(msk_dir / "full.png", mask_array([(0, 0, W, H - 1)]))

    textured(104).save(img_dir / "mismatch.png")
    save_mask(msk_dir / "mismatch.png", mask_array([(2, 2, 10, 10)], w=24, h=16))

    textured(105).save(img_dir / "only_image.png")                  # no mask
    save_mask(msk_dir / "only_mask.png", mask_array([(2, 2, 10, 10)]))  # no image

    (img_dir / "dup_of_g00.png").write_bytes((img_dir / "g00.png").read_bytes())
    save_mask(msk_dir / "dup_of_g00.png", mask_array([(10, 8, 20, 18)]))

    textured(106).save(img_dir / "two.png")
    save_mask(msk_dir / "two.png", mask_array([(2, 2, 8, 8), (30, 20, 40, 28)]))
    return root


@pytest.fixture()
def synthetic(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "ds")
