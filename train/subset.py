"""A fixed random subset of a dataset's training images (for equal-size training): ``python -m train.subset``.

The subset is drawn once from the train part of the fixed split with a seeded generator, saved to
``outputs/splits/<key>_train_subset_<size>_seed<seed>.json`` and committed, so every run and every seed uses the same
images. Validation and test parts are never subsampled.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from audit.load import natural_key

from .config import TrainConfig, load_train_config

log = logging.getLogger("train")


def subset_path(cfg: TrainConfig, key: str, size: int, seed: int) -> Path:
    return cfg.splits_dir / f"{key}_train_subset_{size}_seed{seed}.json"


def configured_subsets(cfg: TrainConfig) -> Dict[str, dict]:
    """``training.train_subset`` of the config: {dataset key: {size, seed}}."""
    return cfg.section("training").get("train_subset", {}) or {}


def select_subset(train_files: List[str], size: int, seed: int) -> List[str]:
    """``size`` file names drawn without replacement from the sorted train files (deterministic)."""
    ordered = sorted(train_files, key=natural_key)
    if not 0 < size <= len(ordered):
        raise ValueError(f"subset size {size} must be between 1 and the {len(ordered)} training images")
    idx = np.random.default_rng(seed).choice(len(ordered), size=size, replace=False)
    return sorted((ordered[i] for i in idx), key=natural_key)


def make_train_subset(cfg: TrainConfig, key: str, size: int, seed: int) -> Path:
    """Write the subset file for one dataset and return its path."""
    split = json.loads((cfg.splits_dir / f"{key}_split.json").read_text(encoding="utf-8"))
    files = select_subset(split["splits"]["train"], size, seed)
    body = {"dataset": cfg.audit.datasets[key].name, "key": key, "size": size, "seed": seed,
            "method": "numpy default_rng(seed).choice without replacement from the naturally sorted train file names",
            "n_train_available": len(split["splits"]["train"]), "images": files}
    path = subset_path(cfg, key, size, seed)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create the fixed training subsets listed in training.train_subset.")
    ap.add_argument("--config", default="train_config.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_train_config(args.config)
    subsets = configured_subsets(cfg)
    if not subsets:
        log.error("no training.train_subset configured in %s", args.config)
        return 2
    for key, spec in subsets.items():
        path = make_train_subset(cfg, key, int(spec["size"]), int(spec["seed"]))
        log.info("wrote %s", path.relative_to(cfg.base_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
