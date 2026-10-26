"""Result tables embedded in README.md, generated from the committed CSV/JSON files.

README.md contains blocks between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->``. ``python -m train.readme_tables``
checks that they match the result files (exit code 1 otherwise); ``--write`` rewrites them. A test runs the check, so the
README cannot drift away from the numbers in ``results/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import TrainConfig, load_train_config
from .paper_figures import size_bins

ROOT = Path(__file__).resolve().parents[1]
FIRST = ["kvasir_test", "cvc_external", "cvc_test", "kvasir_external"]      # E1 ... E4 of the first experiment


def _table(rows: list, headers: list) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _names(cfg: TrainConfig) -> Dict[str, str]:
    return {k: v.name for k, v in cfg.audit.datasets.items()}


def _n_train(cfg: TrainConfig, source: str) -> int:
    return int(json.loads(next(cfg.runs_dir.glob(f"{source}_seed*/run.json")).read_text(encoding="utf-8"))["n_train"])


def exp1_results(cfg: TrainConfig) -> str:
    """E1-E4 of the first experiment: mean ± SD over seeds for every metric."""
    ext = cfg.base_dir / "results" / "extended"
    mbc = pd.read_csv(ext / "metrics_by_condition.csv")
    names = _names(cfg)
    rows = []
    for i, es in enumerate(FIRST):
        g = mbc[mbc["eval_set"] == es]
        src = {"kvasir_test": "kvasir", "cvc_external": "kvasir", "cvc_test": "cvc", "kvasir_external": "cvc"}[es]
        target = es.rsplit("_", 1)[0]
        cells = []
        for m, d in (("dice", 3), ("iou", 3), ("precision", 3), ("recall", 3), ("hd95", 1), ("boundary_iou", 3), ("miss", 3)):
            r = g[g["metric"] == m].iloc[0]
            cells.append(_pm(r["mean_over_seeds"], r["sd_over_seeds"], d))
        flag = " (under-trained, E10)" if src == "cvc" else ""
        n_img = int(g["n_images"].iloc[0])
        rows.append([f"E{i + 1}", f"{names[src]} ({_n_train(cfg, src)}){flag}", f"{names[target]} {'test' if es.endswith('_test') else 'whole set'} ({n_img})"] + cells)
    seeds = int(mbc["n_seeds"].max())
    return (_table(rows, ["ID", "Trained on (images)", "Evaluated on (images)", "Dice", "IoU", "Precision", "Recall", "HD95 [px]", "Boundary IoU", "Miss rate"])
            + f"\n\nMean ± sample SD over {seeds} seeds; each seed value is the mean over the images of the evaluation set.")


def exp1_gap(cfg: TrainConfig) -> str:
    """Generalization gap G = Dice(own test) - Dice(external) with both confidence intervals and the decision rule."""
    ext = cfg.base_dir / "results" / "extended"
    gap = pd.read_csv(ext / "gap_summary.csv")
    dec = pd.read_csv(ext / "decision_rule.csv")
    rows = []
    for _, r in gap[gap["metric"] == "dice"].iterrows():
        d = dec[dec["direction"] == r["direction"]].set_index("delta")["decision_t_ci"]
        rows.append([r["direction"] + (" (E10)" if isinstance(r["status"], str) and r["status"] else ""), f"{r['own_mean']:.3f}", f"{r['external_mean']:.3f}", f"{r['mean_gap']:.3f}", f"{r['sd_gap']:.3f}",
                     f"[{r['ci_t_low']:.3f}, {r['ci_t_high']:.3f}]", f"[{r['ci_boot_low']:.3f}, {r['ci_boot_high']:.3f}]", d[0.05], d[0.03], d[0.10]])
    return _table(rows, ["Model", "Own test", "External", "Gap G", "SD of G", "95% CI (t, df = 2)", "95% CI (cluster bootstrap)", "δ = 0.05", "δ = 0.03", "δ = 0.10"])


def exp1_errors(cfg: TrainConfig) -> str:
    """Quantitative failure indicators: empty predictions, size dependence and the share of poorly segmented frames."""
    ext = cfg.base_dir / "results" / "extended"
    mbc = pd.read_csv(ext / "metrics_by_condition.csv")
    sp = pd.read_csv(ext / "spearman.csv")
    sel = json.loads((cfg.base_dir / "error_analysis" / "selection.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ext / "per_image_metrics_extended.csv", dtype={"sequence_id": str})
    b = size_bins(df)
    lines = [f"- Kvasir-SEG-trained model on CVC-ClinicDB (E2): {sel['frames_below_threshold']} of {sel['total_frames']} frames have seed-averaged Dice < {sel['dice_threshold']}."]
    miss = {es: mbc[(mbc["eval_set"] == es) & (mbc["metric"] == "miss")].iloc[0] for es in FIRST}
    lines.append("- Share of images with an empty prediction (miss rate): " + ", ".join(f"E{i + 1} {miss[es]['mean_over_seeds'] * 100:.1f}%" for i, es in enumerate(FIRST)) + ".")
    for _, r in sp.iterrows():
        if r["variable"] == "gt_area_fraction":
            lines.append(f"- Spearman ρ between per-image Dice and ground-truth polyp area, {r['trained_on']} on {r['evaluated_on']}: {r['spearman_rho']:.2f} (n = {int(r['n_images'])}; descriptive).")
    rows = []
    for i, es in enumerate(FIRST):
        g = df[df["eval_set"] == es]
        area = g["gt_area"]
        cut = np.where(area < b["q1"], "small", np.where(area > b["q3"], "large", "medium"))
        row = [f"E{i + 1}"]
        for name in ("small", "medium", "large"):
            sub = g[cut == name]
            per_seed = sub.groupby("seed")["dice"].mean()
            row.append(f"{per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f} (n = {sub['stem'].nunique()})")
        rows.append(row)
    table = _table(rows, ["ID", f"small (< {b['q1'] * 100:.1f}% of image)", f"medium ({b['q1'] * 100:.1f}-{b['q3'] * 100:.1f}%)", f"large (> {b['q3'] * 100:.1f}%)"])
    return "\n".join(lines) + "\n\nDice by ground-truth polyp size (bins are quartiles of the area at 256 px over all evaluated images; mean ± SD over seeds; n = images):\n\n" + table


def equal_pilot(cfg_equal: TrainConfig) -> str:
    """Runs of the equal-size experiment finished so far, with their completion status."""
    names = _names(cfg_equal)
    rows = []
    for run in sorted(p for p in cfg_equal.runs_dir.iterdir() if (p / "eval_summary.json").is_file()):
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        ev = json.loads((run / "eval_summary.json").read_text(encoding="utf-8"))["sets"]
        src = meta["source"]
        own = ev[f"{src}_test"]
        other = [k for k in names if k != src][0]
        ext = ev[f"{other}_external"]
        rows.append([f"{names[src]}", meta["seed"], meta["n_train"], meta["epochs_run"], meta["best_epoch"], f"{meta['best_val_dice']:.3f}",
                     f"{own['mean_dice']:.3f} ({own['n_images']})", f"{ext['mean_dice']:.3f} ({ext['n_images']})"])
    planned = len(cfg_equal.sources) * len(cfg_equal.seeds)
    return (_table(rows, ["Trained on", "Seed", "Train images", "Epochs", "Best epoch", "Val Dice", "Own test Dice (images)", "External Dice (images)"])
            + f"\n\n**Status: {len(rows)} of {planned} planned runs finished.** Single-seed values; no confidence intervals can be computed yet.")


def build_blocks() -> Dict[str, str]:
    cfg = load_train_config(ROOT / "train_config.yaml")
    cfg_equal = load_train_config(ROOT / "configs" / "equal_size.yaml")
    return {"exp1_results": exp1_results(cfg), "exp1_gap": exp1_gap(cfg), "exp1_errors": exp1_errors(cfg), "equal_pilot": equal_pilot(cfg_equal)}


def apply_blocks(text: str, blocks: Dict[str, str]) -> str:
    """Replace the content between the BEGIN/END markers of each block."""
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", re.S)
        if not pattern.search(text):
            raise ValueError(f"README.md has no block '{name}'")
        text = pattern.sub(lambda m: m.group(1) + body + m.group(2), text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check (or rewrite with --write) the result tables embedded in README.md.")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = ROOT / "README.md"
    old = path.read_text(encoding="utf-8")
    new = apply_blocks(old, build_blocks())
    if args.write:
        path.write_text(new, encoding="utf-8")
        return 0
    if new != old:
        print("README.md tables are out of date; run `python -m train.readme_tables --write`", file=sys.stderr)
        return 1
    print("README.md tables match the result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
