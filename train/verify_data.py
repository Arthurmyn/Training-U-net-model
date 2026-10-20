"""Check that the raw data on this machine is identical to the data the audit and the splits were made from.

``python -m train.verify_data`` compares the MD5 of every image file with ``outputs/per_image_stats_<key>.csv``
(written by the audit on the original machine), and checks that all files named in the split lists exist.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd
from tqdm import tqdm

from audit import checks, load

from .config import load_train_config

log = logging.getLogger("train")


def verify(cfg, progress: bool = True) -> bool:
    """Return True if every dataset matches the audit record. Prints a line per dataset."""
    ok_all = True
    for key, ds in cfg.audit.datasets.items():
        table_path = cfg.audit.output_dir / f"per_image_stats_{key}.csv"
        if not table_path.is_file():
            print(f"[FAIL] {ds.name}: {table_path.name} not found (copy the outputs/ folder from the repository)")
            ok_all = False
            continue
        ref = pd.read_csv(table_path).set_index("image_file")["image_md5"].to_dict()
        files = load.discover(ds)
        on_disk = {p.name: p for p in files.images.values()}
        missing = sorted(set(ref) - set(on_disk))
        extra = sorted(set(on_disk) - set(ref))
        masks_missing = sorted(s for s in files.images if s not in files.masks)
        bad = []
        for name in tqdm(sorted(set(ref) & set(on_disk)), desc=ds.name, disable=not progress, mininterval=2.0):
            if checks.file_md5(on_disk[name]) != ref[name]:
                bad.append(name)
        split = json.loads((cfg.splits_dir / f"{key}_split.json").read_text(encoding="utf-8"))
        listed = {f for part in split["splits"].values() for f in part}
        unlisted = sorted(listed - set(on_disk))
        ok = not (missing or extra or bad or masks_missing or unlisted) and len(ref) == ds.expected_count
        ok_all &= ok
        print(f"[{'OK' if ok else 'FAIL'}] {ds.name}: {len(on_disk)} images on disk, {len(ref)} in the audit record, "
              f"{len(bad)} with different content, {len(missing)} missing, {len(extra)} extra, {len(masks_missing)} without mask, "
              f"{len(unlisted)} split entries not found")
        for label, items in (("different content", bad), ("missing", missing), ("extra", extra)):
            if items:
                print(f"       {label}: {', '.join(items[:5])}{' ...' if len(items) > 5 else ''}")
    return ok_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify that the raw data matches the audit record.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    ok = verify(load_train_config(args.config), progress=not args.no_progress)
    print("DATA OK: safe to train." if ok else "DATA PROBLEM: do not train until this is fixed (see docs/SETUP_ru.md).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
