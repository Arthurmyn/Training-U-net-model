"""Configuration loading. All paths in the YAML file are relative to the file's folder."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class SequenceMeta:
    """Where to find video-sequence IDs for a dataset."""

    file: str
    stem_column: str
    sequence_column: str


@dataclass(frozen=True)
class ReviewedPair:
    """A near-duplicate candidate pair that a person has looked at (stems, no extension)."""

    a: str
    b: str
    verdict: str          # "duplicate" (keep in one split) or "different_images" (pHash false positive)
    note: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    """One dataset: folders, expectations and split recipe."""

    key: str
    name: str
    root: Path
    image_dir: str
    mask_dir: str
    expected_count: int
    split_method: str
    split_sizes: Dict[str, int] = field(default_factory=dict)
    split_fractions: Dict[str, float] = field(default_factory=dict)
    expected_sequences: Optional[int] = None
    sequence_meta: Optional[SequenceMeta] = None
    crosscheck_image_dir: Optional[str] = None
    crosscheck_mask_dir: Optional[str] = None
    reviewed_pairs: Tuple[ReviewedPair, ...] = ()

    @property
    def image_path(self) -> Path:
        return self.root / self.image_dir

    @property
    def mask_path(self) -> Path:
        return self.root / self.mask_dir


@dataclass(frozen=True)
class AuditConfig:
    """Whole-run configuration."""

    seed: int
    output_dir: Path
    mask_binarize: int
    full_mask_fraction: float
    near_duplicate_hamming: int
    datasets: Dict[str, DatasetConfig]
    base_dir: Path

    def settings_dict(self) -> dict:
        """Config values that go into summary.json (no absolute paths)."""
        return {
            "seed": self.seed,
            "mask_binarize_threshold": self.mask_binarize,
            "full_mask_fraction": self.full_mask_fraction,
            "near_duplicate_hamming": self.near_duplicate_hamming,
        }


def load_config(path: str | Path) -> AuditConfig:
    """Read a YAML config; relative paths resolve against the config file's folder."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    base = path.parent
    thr = raw.get("thresholds", {})
    datasets: Dict[str, DatasetConfig] = {}
    for key, d in raw["datasets"].items():
        split = d.get("split", {})
        seq = d.get("sequence_metadata")
        xc = d.get("crosscheck") or {}
        datasets[key] = DatasetConfig(
            key=key,
            name=d.get("name", key),
            root=(base / d["root"]).resolve(),
            image_dir=d["image_dir"],
            mask_dir=d["mask_dir"],
            expected_count=int(d["expected_count"]),
            split_method=split.get("method", "random"),
            split_sizes={k: int(v) for k, v in split.get("sizes", {}).items()},
            split_fractions={k: float(v) for k, v in split.get("fractions", {}).items()},
            expected_sequences=d.get("expected_sequences"),
            sequence_meta=SequenceMeta(seq["file"], str(seq["stem_column"]), str(seq["sequence_column"])) if seq else None,
            crosscheck_image_dir=xc.get("image_dir"),
            crosscheck_mask_dir=xc.get("mask_dir"),
            reviewed_pairs=tuple(ReviewedPair(str(r["a"]), str(r["b"]), r["verdict"], r.get("note", ""))
                                 for r in d.get("reviewed_near_duplicates", [])),
        )
    return AuditConfig(
        seed=int(raw.get("seed", 42)),
        output_dir=(base / raw.get("output_dir", "outputs")).resolve(),
        mask_binarize=int(thr.get("mask_binarize", 127)),
        full_mask_fraction=float(thr.get("full_mask_fraction", 0.90)),
        near_duplicate_hamming=int(thr.get("near_duplicate_hamming", 4)),
        datasets=datasets,
        base_dir=base,
    )
