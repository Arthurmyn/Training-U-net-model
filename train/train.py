"""Train one model: ``python -m train.train --source kvasir --seed 1``."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_train_config
from .engine import train_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train one U-Net (source dataset, seed).")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--source", required=True, help="dataset key to train on (kvasir or cvc)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--no-resume", action="store_true", help="discard an unfinished run and start again")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="force the device (e.g. cuda: stop with an error instead of silently using the CPU)")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_train_config(args.config)
    if args.device:
        cfg = cfg.override(**{"training.device": args.device})
    train_run(cfg, args.source, args.seed, resume=not args.no_resume, progress=not args.no_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
