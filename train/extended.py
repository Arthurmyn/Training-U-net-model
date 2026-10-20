"""Extended analysis driver: ``python -m train.extended <stage>``.

Stages: ``inference`` (re-run the saved weights, extended per-image metrics, acceptance checks A and B),
``stats`` (metrics by condition, gaps, decision rule, Spearman), ``figures``, ``errors`` (error-analysis material),
``report`` (results/extended/README.md and METRICS.md) or ``all``. Nothing under runs/, outputs/ or the original
results files is modified; new files go to results/extended/ and error_analysis/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import extended_metrics as em
from .config import TrainConfig, load_train_config
from .data import SegDataset
from .evaluate import build_eval_sets
from .metrics import dice_per_image
from .model import UNet

log = logging.getLogger("train")

EXISTING_COLUMNS = ["run", "source", "seed", "eval_set", "kind", "target", "dice", "gt_area", "pred_area", "stem",
                    "image_file", "sequence_id", "target_split"]
NEW_COLUMNS = ["tp", "fp", "fn", "iou", "precision", "recall", "hd95", "boundary_iou", "miss"]
EXTENDED_FILE = "per_image_metrics_extended.csv"


class AcceptanceError(RuntimeError):
    """Raised when the recomputed metrics do not reproduce the original evaluation."""


def ext_section(cfg: TrainConfig) -> dict:
    return cfg.section("extended")


def out_dir(cfg: TrainConfig) -> Path:
    return (cfg.base_dir / ext_section(cfg).get("output_dir", "results/extended")).resolve()


def error_dir(cfg: TrainConfig) -> Path:
    return (cfg.base_dir / ext_section(cfg).get("error_dir", "error_analysis")).resolve()


def load_model(run_dir: Path, base_channels: int, device: torch.device) -> UNet:
    """The best checkpoint (by validation Dice) of a run, in eval mode."""
    model = UNet(3, base_channels).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    return model.eval()


@torch.no_grad()
def predict_masks(model: UNet, ds: SegDataset, device: torch.device, batch_size: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Binary predictions (n, H, W) and the logits-derived Dice of the existing code (n,), no augmentation."""
    preds, dices = [], []
    for start in range(0, len(ds), batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        img = torch.stack([b[0] for b in batch]).to(device)
        mask = torch.stack([b[1] for b in batch])
        logits = model(img).float().cpu()
        preds.append((torch.sigmoid(logits) > threshold)[:, 0].numpy())
        dices.append(dice_per_image(logits, mask, threshold).numpy())
    return np.concatenate(preds), np.concatenate(dices)


def run_inference(cfg: TrainConfig, progress: bool = True) -> pd.DataFrame:
    """Extended metrics for every run and every evaluation set (same sets, preprocessing and threshold as the original)."""
    e = ext_section(cfg)
    device = torch.device(e.get("device", "cpu"))
    batch = int(e.get("batch_size", 16))
    d = float(e.get("boundary_iou_d", 5))
    thr = float(cfg.section("training").get("threshold", 0.5))
    base = int(cfg.section("model").get("base_channels", 32))
    rows: List[dict] = []
    runs = sorted(p.parent for p in cfg.runs_dir.glob("*/run.json"))
    if not runs:
        raise FileNotFoundError(f"no runs found under {cfg.runs_dir}")
    for run_dir in runs:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if meta.get("status") != "finished" or not (run_dir / "best.pt").is_file():
            raise FileNotFoundError(f"{run_dir.name}: unfinished run or best.pt missing")
        source, seed = meta["source"], int(meta["seed"])
        model = load_model(run_dir, base, device)
        for name, kind, part, items, own_split in build_eval_sets(cfg, source):
            ds = SegDataset(items, cfg.image_size, cfg.mask_threshold, None, False)
            preds, torch_dice = predict_masks(model, ds, device, batch, thr)
            gts = ds.masks.astype(bool)
            for i, it in enumerate(tqdm(items, desc=f"{run_dir.name} {name}", disable=not progress, mininterval=3.0)):
                m = em.all_metrics(preds[i], gts[i], d)
                assert abs(m["dice"] - float(torch_dice[i])) < 1e-5, "count-based Dice differs from dice_per_image"
                rows.append({"run": run_dir.name, "source": source, "seed": seed, "eval_set": name, "kind": kind,
                             "target": name.rsplit("_", 1)[0], "dice": m["dice"], "gt_area": float(gts[i].mean()),
                             "pred_area": float(preds[i].mean()), "stem": it.stem, "image_file": it.image_file,
                             "sequence_id": it.sequence_id, "target_split": own_split.get(it.stem, part),
                             **{k: m[k] for k in NEW_COLUMNS}})
    return pd.DataFrame(rows, columns=EXISTING_COLUMNS + NEW_COLUMNS)


def acceptance_checks(new: pd.DataFrame, old: pd.DataFrame, mean_tol: float = 0.002, image_tol: float = 0.01,
                      share_required: float = 0.99, iou_tol: float = 1e-4) -> dict:
    """Check A: recomputed Dice reproduces the original per-image Dice. Check B: iou == dice / (2 - dice)."""
    key = ["run", "eval_set", "stem"]
    m = new[key + ["dice"]].merge(old[key + ["dice"]], on=key, how="outer", suffixes=("_new", "_old"), indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"].copy()
    m["abs_diff"] = (m["dice_new"] - m["dice_old"]).abs()

    def summarize(g: pd.DataFrame) -> dict:
        return {"n": int(len(g)), "mean_abs_diff": float(g["abs_diff"].mean()), "max_abs_diff": float(g["abs_diff"].max()),
                "share_below_0.01": float((g["abs_diff"] < image_tol).mean())}

    by_set = {k: summarize(g) for k, g in m.groupby("eval_set")}
    by_run_set = {f"{r}/{e}": summarize(g) for (r, e), g in m.groupby(["run", "eval_set"])}
    a_ok = unmatched == 0 and all(v["mean_abs_diff"] < mean_tol and v["share_below_0.01"] >= share_required for v in by_set.values())
    b_dev = float((new["iou"] - new["dice"] / (2 - new["dice"])).abs().max())
    return {"A_dice_reproduces_original": {"passed": bool(a_ok), "unmatched_rows": unmatched, "by_eval_set": by_set,
                                           "by_run_and_eval_set": by_run_set,
                                           "criteria": f"per-eval-set mean abs diff < {mean_tol} and >= {share_required:.0%} of images with diff < {image_tol}"},
            "B_iou_equals_dice_over_2_minus_dice": {"passed": bool(b_dev < iou_tol), "max_abs_deviation": b_dev, "tolerance": iou_tol}}


def write_acceptance(cfg: TrainConfig, result: dict) -> None:
    out = out_dir(cfg)
    (out / "acceptance_checks.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_inference(cfg: TrainConfig, progress: bool = True) -> pd.DataFrame:
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df = run_inference(cfg, progress)
    df.to_csv(out / EXTENDED_FILE, index=False, float_format="%.6f")
    old = pd.concat([pd.read_csv(p, dtype={"sequence_id": str}) for p in sorted(cfg.runs_dir.glob("*/per_image_dice.csv"))], ignore_index=True)
    result = acceptance_checks(df, old)
    write_acceptance(cfg, result)
    a, b = result["A_dice_reproduces_original"], result["B_iou_equals_dice_over_2_minus_dice"]
    for k, v in a["by_eval_set"].items():
        log.info("check A %-16s n=%4d  mean |diff| %.5f  max %.4f  share<0.01 %.3f", k, v["n"], v["mean_abs_diff"], v["max_abs_diff"], v["share_below_0.01"])
    log.info("check A %s | check B %s (max deviation %.2e)", "PASSED" if a["passed"] else "FAILED", "PASSED" if b["passed"] else "FAILED", b["max_abs_deviation"])
    if not (a["passed"] and b["passed"]):
        raise AcceptanceError("acceptance check failed; see results/extended/acceptance_checks.json. Later stages were not run.")
    return df


def load_extended(cfg: TrainConfig) -> pd.DataFrame:
    path = out_dir(cfg) / EXTENDED_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found; run `python -m train.extended inference` first")
    return pd.read_csv(path, dtype={"sequence_id": str})


STAGES = ("inference", "stats", "figures", "errors", "report")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extended metrics, statistics, figures and error-analysis material.")
    ap.add_argument("stage", choices=STAGES + ("all",))
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_train_config(args.config)
    todo = STAGES if args.stage == "all" else (args.stage,)
    for stage in todo:
        if stage == "inference":
            try:
                stage_inference(cfg, progress=not args.no_progress)
            except AcceptanceError as exc:
                log.error("%s", exc)
                return 3
        else:
            import importlib
            module = {"stats": "extended_stats", "figures": "extended_figures", "errors": "error_material", "report": "extended_report"}[stage]
            importlib.import_module(f"train.{module}").run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
