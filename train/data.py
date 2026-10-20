"""Datasets built from the fixed split files written by the audit (outputs/splits/)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from audit import load, preprocess, splits as audit_splits

from .augment import augment
from .config import TrainConfig

log = logging.getLogger("train")


@dataclass(frozen=True)
class Item:
    """One image/mask pair of a dataset."""

    dataset: str          # config key, e.g. "kvasir"
    stem: str
    image_path: Path
    mask_path: Path
    image_file: str
    sequence_id: Optional[str] = None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m audit.run --config config.yaml` first or copy outputs/splits/.")
    return json.loads(path.read_text(encoding="utf-8"))


def _sequence_ids(cfg: TrainConfig, key: str) -> Dict[str, str]:
    csv = cfg.splits_dir / f"{key}_split.csv"
    out: Dict[str, str] = {}
    if csv.is_file():
        import pandas as pd
        df = pd.read_csv(csv, dtype=str)
        if "sequence_id" in df.columns:
            out = {Path(f).stem: s for f, s in zip(df["image_file"], df["sequence_id"])}
    return out


def load_items(cfg: TrainConfig, key: str) -> Tuple[Dict[str, List[Item]], List[Item]]:
    """Return ``({part: items}, all_items)`` for one dataset, from its fixed split and external-test files.

    The split must be disjoint and cover the external-test list exactly; this is asserted.
    """
    ds = cfg.audit.datasets[key]
    files = load.discover(ds)
    seq = _sequence_ids(cfg, key)

    def make(file_name: str) -> Item:
        stem = Path(file_name).stem
        if stem not in files.images or stem not in files.masks:
            raise FileNotFoundError(f"{ds.name}: {file_name} listed in the split but missing on disk ({ds.image_path})")
        return Item(key, stem, files.images[stem], files.masks[stem], files.images[stem].name, seq.get(stem))

    split = _read_json(cfg.splits_dir / f"{key}_split.json")
    ext = _read_json(cfg.splits_dir / f"{key}_external_test.json")
    parts = {name: [make(f) for f in names] for name, names in split["splits"].items()}
    all_items = [make(f) for f in ext["images"]]
    audit_splits.validate_split({k: [i.stem for i in v] for k, v in parts.items()}, [i.stem for i in all_items])
    return parts, all_items


class SegDataset(Dataset):
    """Preprocessed image/mask pairs held in memory as uint8 (about 160 MB for 800 images at 256x256).

    Preprocessing is the audited one (audit/preprocess.py): resize, bilinear for images, nearest for masks,
    mask > threshold, pixels scaled to [0, 1]. ``augmentation`` (a dict) is applied on the fly, training only.
    """

    def __init__(self, items: List[Item], size: int, threshold: int, augmentation: Optional[dict] = None,
                 progress: bool = False) -> None:
        self.items = list(items)
        self.augmentation = augmentation
        n = len(self.items)
        self.images = np.zeros((n, size, size, 3), np.uint8)
        self.masks = np.zeros((n, size, size), np.uint8)
        for i, it in enumerate(tqdm(self.items, desc="loading", disable=not progress, mininterval=2.0)):
            img, mask = preprocess.preprocess_pair(it.image_path, it.mask_path, size, threshold)
            self.images[i] = np.rint(img * 255.0).astype(np.uint8)       # exact: img is k / 255
            self.masks[i] = mask

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(self.masks[i]).unsqueeze(0).float()
        if self.augmentation:
            img, mask = augment(img, mask, self.augmentation)
        return img, mask, i


def limited(items: List[Item], limit: Optional[int]) -> List[Item]:
    """First ``limit`` items (deterministic), used by the quick smoke test."""
    return items if not limit else items[:limit]
