"""Read-only discovery of files, tree printing, sequence metadata and pixel loading."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import DatasetConfig

log = logging.getLogger("audit")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def natural_key(text: str) -> List[object]:
    """Sort key so that '2' < '10' < '100' (CVC file names are numeric)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def list_images(folder: Path) -> Tuple[Dict[str, Path], List[str]]:
    """Map stem -> path for image files in ``folder``.

    Returns ``(files, ambiguous_stems)``. If two files share a stem (e.g. a.jpg and a.png)
    the first in sorted order is kept and the stem is reported as ambiguous.
    """
    files: Dict[str, Path] = {}
    ambiguous: List[str] = []
    if not folder.is_dir():
        return files, ambiguous
    for p in sorted((q for q in folder.iterdir() if q.is_file() and not q.name.startswith(".")),
                    key=lambda q: natural_key(q.name)):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if p.stem in files:
            ambiguous.append(p.stem)
            continue
        files[p.stem] = p
    return files, ambiguous


@dataclass
class Discovery:
    """Result of matching images to masks by filename stem."""

    images: Dict[str, Path]
    masks: Dict[str, Path]
    matched: List[str]
    unmatched_images: List[str]
    unmatched_masks: List[str]
    ambiguous_stems: List[str] = field(default_factory=list)
    xcheck_images: Dict[str, Path] = field(default_factory=dict)
    xcheck_masks: Dict[str, Path] = field(default_factory=dict)


def discover(cfg: DatasetConfig) -> Discovery:
    """Find images and masks and match them by stem (natural order, deterministic)."""
    images, amb_i = list_images(cfg.image_path)
    masks, amb_m = list_images(cfg.mask_path)
    matched = sorted(set(images) & set(masks), key=natural_key)
    xi: Dict[str, Path] = {}
    xm: Dict[str, Path] = {}
    if cfg.crosscheck_image_dir and cfg.crosscheck_mask_dir:
        xi, _ = list_images(cfg.root / cfg.crosscheck_image_dir)
        xm, _ = list_images(cfg.root / cfg.crosscheck_mask_dir)
    return Discovery(
        images=images,
        masks=masks,
        matched=matched,
        unmatched_images=sorted(set(images) - set(masks), key=natural_key),
        unmatched_masks=sorted(set(masks) - set(images), key=natural_key),
        ambiguous_stems=sorted(set(amb_i + amb_m), key=natural_key),
        xcheck_images=xi,
        xcheck_masks=xm,
    )


def tree_lines(root: Path, max_depth: int = 4, examples: int = 2) -> List[str]:
    """Compact directory tree: each folder with file count, extensions and a few example names."""
    lines = [f"{root.name}/"]

    def walk(folder: Path, prefix: str, depth: int) -> None:
        entries = sorted(folder.iterdir(), key=lambda q: natural_key(q.name))
        dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
        files = [e for e in entries if e.is_file() and not e.name.startswith(".")]
        if files:
            exts = sorted({f.suffix.lower() for f in files})
            ex = ", ".join(f.name for f in files[:examples])
            lines.append(f"{prefix}[{len(files)} files {'/'.join(exts)}] e.g. {ex}")
        for i, d in enumerate(dirs):
            last = i == len(dirs) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{d.name}/")
            if depth < max_depth:
                walk(d, prefix + ("    " if last else "│   "), depth + 1)

    walk(root, "", 1)
    return lines


def load_sequence_ids(cfg: DatasetConfig) -> Tuple[Dict[str, str], str]:
    """Read stem -> sequence ID from the metadata file.

    Returns ``(mapping, note)``. The mapping is empty if no sequence information is available;
    ``note`` explains why.
    """
    meta = cfg.sequence_meta
    if meta is None:
        return {}, "no sequence metadata configured"
    path = cfg.root / meta.file
    if not path.is_file():
        return {}, f"sequence metadata file '{meta.file}' not found"
    df = pd.read_csv(path, dtype=str)
    for col in (meta.stem_column, meta.sequence_column):
        if col not in df.columns:
            return {}, f"column '{col}' missing in '{meta.file}'"
    df = df.dropna(subset=[meta.stem_column, meta.sequence_column])
    mapping = {str(s).strip(): str(q).strip() for s, q in zip(df[meta.stem_column], df[meta.sequence_column])}
    return mapping, f"read from '{meta.file}'"


def open_image_rgb(path: Path) -> Image.Image:
    """Open and fully decode an image; raises on corrupt or truncated files."""
    with Image.open(path) as im:
        im.load()
        n_bands = len(im.getbands())
        mode = im.mode
        rgb = im.convert("RGB")
    rgb.info["orig_mode"] = mode
    rgb.info["orig_bands"] = n_bands
    return rgb


def read_mask_raw(path: Path) -> Tuple[np.ndarray, str]:
    """Decode a mask to a 2-D array of raw values; returns ``(array, original_mode)``.

    Grayscale masks are used as stored. Multi-channel masks (e.g. RGB PNG) use the maximum
    over channels, so a coloured polyp label is not lost.
    """
    with Image.open(path) as im:
        im.load()
        mode = im.mode
        if mode in ("L", "I", "I;16", "F"):
            arr = np.array(im)
        elif mode == "1":
            arr = np.array(im.convert("L"))
        else:
            arr = np.array(im.convert("RGB")).max(axis=2)
    return arr, mode
