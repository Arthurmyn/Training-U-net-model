"""Robustness check: metrics at the ORIGINAL image resolution instead of 256 x 256.

The models are trained and evaluated at 256 x 256 after resizing every image to a square (this changes the aspect ratio).
Here the predicted probability map is resized back to the size of the original image (bilinear), thresholded at 0.5 and
compared with the ORIGINAL ground-truth mask (value > 127). Weights, splits and threshold are those of the original
evaluation; nothing is retrained. If the conclusions do not change, the 256 x 256 protocol does not distort them.

    python -m train.original_resolution --config configs/equal_size.yaml

Writes ``results/original_resolution/`` (per-image CSV, comparison tables, README). HD95 and Boundary IoU are not
recomputed here (they are slow at full resolution and are defined in pixels of the 256 x 256 grid).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from audit.load import read_mask_raw

from . import extended_metrics as em
from .config import TrainConfig, load_train_config
from .data import SegDataset
from .evaluate import build_eval_sets
from .extended import get_device, load_model

log = logging.getLogger("train")
OUT_NAME = "original_resolution"


def out_dir(cfg: TrainConfig) -> Path:
    return cfg.results_dir / OUT_NAME


def upsample_to_original(prob: torch.Tensor, height: int, width: int, threshold: float) -> np.ndarray:
    """Probability map (1, h, w) of the 256 x 256 grid -> boolean mask of the original size (bilinear, then threshold)."""
    big = F.interpolate(prob.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    return (big[0, 0] > threshold).cpu().numpy()


def image_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    c = em.confusion_counts(pred, gt)
    return {"dice": em.dice_from_counts(**c), "iou": em.iou_from_counts(**c), "precision": em.precision_from_counts(c["tp"], c["fp"]),
            "recall": em.recall_from_counts(c["tp"], c["fn"]), "miss": int(not pred.any())}


class SetCache:
    """Preprocessed images and original masks of an evaluation set, decoded once and reused by every run."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.sets: Dict[str, Tuple[list, SegDataset, List[np.ndarray], dict]] = {}

    def get(self, source: str, name: str):
        if name not in self.sets:
            for set_name, kind, part, items, own_split in build_eval_sets(self.cfg, source):
                if set_name == name:
                    ds = SegDataset(items, self.cfg.image_size, self.cfg.mask_threshold, None, False)
                    masks = [read_mask_raw(it.mask_path)[0] > self.cfg.mask_threshold for it in items]
                    self.sets[name] = (items, ds, masks, {"kind": kind, "part": part, "own_split": own_split})
        return self.sets[name]


@torch.no_grad()
def run_one(cfg: TrainConfig, run_dir: Path, cache: SetCache, device: torch.device, progress: bool = True) -> pd.DataFrame:
    """Original-resolution metrics of one run on every evaluation set it was evaluated on."""
    import json
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    source, seed = meta["source"], int(meta["seed"])
    thr = float(cfg.section("training").get("threshold", 0.5))
    model = load_model(run_dir, int(cfg.section("model").get("base_channels", 32)), device)
    old = pd.read_csv(run_dir / "per_image_dice.csv", dtype={"sequence_id": str})
    rows = []
    for name in old["eval_set"].unique():
        items, ds, masks, info = cache.get(source, name)
        by_stem = {it.stem: i for i, it in enumerate(items)}
        for start in tqdm(range(0, len(ds), 16), desc=f"{run_dir.name} {name}", disable=not progress, mininterval=5.0):
            idx = list(range(start, min(start + 16, len(ds))))
            x = torch.stack([ds[i][0] for i in idx]).to(device)
            prob = torch.sigmoid(model(x).float()).cpu()
            for k, i in enumerate(idx):
                gt = masks[i]
                m = image_metrics(upsample_to_original(prob[k], gt.shape[0], gt.shape[1], thr), gt)
                rows.append({"run": run_dir.name, "source": source, "seed": seed, "eval_set": name, "kind": info["kind"], "stem": items[i].stem,
                             "image_file": items[i].image_file, "sequence_id": items[i].sequence_id, "width": gt.shape[1], "height": gt.shape[0],
                             **{f"{k2}_original": v for k2, v in m.items()}})
    df = pd.DataFrame(rows)
    ref = old[["eval_set", "stem", "dice"]].rename(columns={"dice": "dice_256"})
    return df.merge(ref, on=["eval_set", "stem"], how="left", validate="many_to_one")


def summarize(cfg: TrainConfig, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Comparison of the 256 x 256 and the original-resolution Dice, and the generalization gap at both resolutions."""
    names = {k: v.name for k, v in cfg.audit.datasets.items()}
    per_seed = df.groupby(["source", "eval_set", "seed"]).agg(dice_256=("dice_256", "mean"), dice_original=("dice_original", "mean"),
                                                             iou_original=("iou_original", "mean"), miss_original=("miss_original", "mean")).reset_index()
    cond = []
    for (src, es), g in per_seed.groupby(["source", "eval_set"]):
        sub = df[(df["source"] == src) & (df["eval_set"] == es)]
        cond.append({"trained_on": names[src], "eval_set": es, "images": int(sub["stem"].nunique()), "seeds": int(g["seed"].nunique()),
                     "dice_256": g["dice_256"].mean(), "dice_original": g["dice_original"].mean(), "difference": g["dice_original"].mean() - g["dice_256"].mean(),
                     "mean_abs_image_difference": float((sub["dice_original"] - sub["dice_256"]).abs().mean()),
                     "max_abs_image_difference": float((sub["dice_original"] - sub["dice_256"]).abs().max()),
                     "iou_original": g["iou_original"].mean(), "miss_rate_original": g["miss_original"].mean()})
    gaps = []
    for src in cfg.sources:
        others = [k for k in cfg.audit.datasets if k != src]
        own, ext = f"{src}_test", f"{others[0]}_external"
        for col, label in (("dice_256", "256 x 256"), ("dice_original", "original resolution")):
            a = per_seed[per_seed["eval_set"] == own].set_index("seed")[col]
            b = per_seed[per_seed["eval_set"] == ext].set_index("seed")[col]
            if a.empty or b.empty:
                continue
            g = (a - b).dropna()
            n = len(g)
            half = stats.t.ppf(0.975, n - 1) * g.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            gaps.append({"trained_on": names[src], "resolution": label, "seeds": n, "own_test": a.mean(), "external": b.mean(), "gap": g.mean(),
                         "ci_t_low": g.mean() - half, "ci_t_high": g.mean() + half})
    return {"by_condition": pd.DataFrame(cond), "gap": pd.DataFrame(gaps)}


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def run(cfg: TrainConfig, progress: bool = True) -> Path:
    """Evaluate every finished run of ``cfg.runs_dir`` at the original resolution and write ``results/original_resolution/``."""
    device = get_device(cfg)
    cache = SetCache(cfg)
    frames = [run_one(cfg, d.parent, cache, device, progress) for d in sorted(cfg.runs_dir.glob("*/run.json"))]
    if not frames:
        raise FileNotFoundError(f"no runs under {cfg.runs_dir}")
    df = pd.concat(frames, ignore_index=True)
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image_metrics_original.csv", index=False, float_format="%.6f")
    tables = summarize(cfg, df)
    tables["by_condition"].to_csv(out / "by_condition.csv", index=False, float_format="%.6f")
    tables["gap"].to_csv(out / "gap.csv", index=False, float_format="%.6f")
    text = f"""# Metrics at the original image resolution

Robustness check of the 256 x 256 protocol. The predicted probability map is resized back to each image's original size
(bilinear), thresholded at 0.5 and compared with the original ground-truth mask. Same weights, splits and threshold as the
original evaluation; nothing was retrained. Generated by `python -m train.original_resolution`.

## Dice by evaluation set (mean over seeds)

{_md(tables['by_condition'])}

`difference` = original-resolution Dice minus 256 x 256 Dice (mean over seeds); the last columns give the per-image absolute difference.

## Generalization gap G = Dice(own test) - Dice(external)

{_md(tables['gap'])}

HD95 and Boundary IoU are not recomputed at full resolution.
"""
    (out / "README.md").write_text(text, encoding="utf-8")
    log.info("wrote %s", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Recompute Dice, IoU, precision and recall at the original image resolution.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="override extended.device")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_train_config(args.config)
    if args.device:
        cfg = cfg.override(**{"extended.device": args.device})
    run(cfg, progress=not args.no_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
