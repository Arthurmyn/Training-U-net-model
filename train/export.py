"""One combined per-image metrics file over all runs: ``python -m train.export`` -> results/per_image_metrics.csv."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .analyze import load_results
from .config import TrainConfig, load_train_config

log = logging.getLogger("train")

COLUMNS = ["direction", "train_dataset", "test_dataset", "condition", "seed", "image", "dice", "gt_area_fraction",
           "pred_area_fraction", "sequence_id", "eval_set", "target_split", "run"]


def build_per_image_metrics(cfg: TrainConfig) -> pd.DataFrame:
    """All runs, one row per (run, evaluation set, image). ``direction`` is "train dataset -> test dataset"."""
    d = load_results(cfg.runs_dir)
    names = {k: v.name for k, v in cfg.audit.datasets.items()}
    out = pd.DataFrame({
        "direction": d["source"].map(names) + " -> " + d["target"].map(names),
        "train_dataset": d["source"].map(names),
        "test_dataset": d["target"].map(names),
        "condition": d["kind"].map({"in_domain": "in-domain", "external": "external"}),
        "seed": d["seed"], "image": d["image_file"], "dice": d["dice"],
        "gt_area_fraction": d["gt_area"], "pred_area_fraction": d["pred_area"],
        "sequence_id": d["sequence_id"], "eval_set": d["eval_set"], "target_split": d["target_split"], "run": d["run"],
    })
    return out[COLUMNS].sort_values(["direction", "eval_set", "seed", "image"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write results/per_image_metrics.csv from all evaluated runs.")
    ap.add_argument("--config", default="train_config.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_train_config(args.config)
    df = build_per_image_metrics(cfg)
    path: Path = cfg.results_dir / "per_image_metrics.csv"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    log.info("wrote %s (%d rows)", path.name, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
