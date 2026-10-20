"""Run the whole experiment: train and evaluate every (source, seed). Safe to stop and start again.

    python -m train.run_all --smoke      # quick test of the setup (a few minutes at most)
    python -m train.run_all              # the full experiment
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from .config import load_train_config
from .engine import environment, select_device, train_run
from .evaluate import evaluate_run

log = logging.getLogger("train")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate all runs of the experiment.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny test run (32 images, 2 epochs) in runs_smoke/, with a time estimate")
    ap.add_argument("--sources", nargs="+", help="only these datasets (default: all in the config)")
    ap.add_argument("--seeds", nargs="+", type=int, help="only these seeds (default: all in the config)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--no-eval", action="store_true", help="train only")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    cfg = load_train_config(args.config)
    limit = None
    if args.smoke:
        limit = 32
        cfg = cfg.override(**{"runs_dir": "runs_smoke", "training.max_epochs": 2, "training.patience": 2})
    sources = args.sources or cfg.sources
    seeds = args.seeds or (cfg.seeds[:1] if args.smoke else cfg.seeds)
    plan = [(s, seed) for seed in seeds for s in sources]
    device = select_device(cfg.section("training").get("device", "auto"))
    print(f"device: {device}  torch {torch.__version__}  runs folder: {cfg.runs_dir.name}/")
    print("plan: " + ", ".join(f"{s}_seed{seed}" for s, seed in plan))
    if args.dry_run:
        return 0

    t0 = time.time()
    for source, seed in plan:
        run_dir = train_run(cfg, source, seed, progress=not args.no_progress, limit=limit)
        if not args.no_eval:
            if (run_dir / "per_image_dice.csv").is_file() and not args.smoke:
                log.info("%s already evaluated, skipping", run_dir.name)
            else:
                evaluate_run(cfg, run_dir, limit=limit, progress=not args.no_progress)
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min")

    rows = []
    for source, seed in plan:
        d = cfg.runs_dir / f"{source}_seed{seed}"
        if (d / "eval_summary.json").is_file():
            meta, ev = json.loads((d / "run.json").read_text(encoding="utf-8")), json.loads((d / "eval_summary.json").read_text(encoding="utf-8"))
            rows.append({"run": d.name, "epochs": meta["epochs_run"], "best_epoch": meta["best_epoch"], "val_dice": round(meta["best_val_dice"], 4),
                         **{k: round(v["mean_dice"], 4) for k, v in ev["sets"].items() if not k.endswith("_val")}})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))

    if args.smoke:
        full = load_train_config(args.config)
        for source, seed in plan:
            m = pd.read_csv(cfg.runs_dir / f"{source}_seed{seed}" / "metrics.csv")
            meta = json.loads((cfg.runs_dir / f"{source}_seed{seed}" / "run.json").read_text(encoding="utf-8"))
            from .data import load_items
            parts, _ = load_items(full, source)
            n_small = meta["n_train"] + meta["n_val"]
            n_full = len(parts["train"]) + len(parts["val"])
            per_epoch = float(m["seconds"].iloc[-1]) * n_full / n_small        # last epoch: after warm-up
            ep = int(full.section("training")["max_epochs"])
            print(f"estimate for {source}: ~{per_epoch:.0f} s per epoch on this machine; up to {ep} epochs = {per_epoch * ep / 3600:.1f} h per run "
                  f"(early stopping usually ends sooner); {len(full.seeds)} seeds x {len(full.sources)} datasets = {len(full.seeds) * len(full.sources)} runs.")
        print("\nSMOKE TEST PASSED. Delete runs_smoke/ when you are done (it is not part of the results).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
