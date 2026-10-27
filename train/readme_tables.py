"""Result tables embedded in README.md, generated from the committed CSV/JSON files of the experiment (``final_results/``).

README.md contains blocks between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->``. ``python -m train.readme_tables``
checks that they match the result files (exit code 1 otherwise); ``--write`` rewrites them. A test runs the check, so the
README cannot drift away from the numbers in the result files.
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
CONDITIONS = ["kvasir_test", "cvc_external", "cvc_test", "kvasir_external"]      # E1 ... E4


def _table(rows: list, headers: list) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _names(cfg: TrainConfig) -> Dict[str, str]:
    return {k: v.name for k, v in cfg.audit.datasets.items()}


def _n_train(cfg: TrainConfig, source: str) -> int:
    return int(json.loads(next(cfg.runs_dir.glob(f"{source}_seed*/run.json")).read_text(encoding="utf-8"))["n_train"])


def _gpu(run_dir: Path) -> str:
    env = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["environment"]
    return env.get("gpu", env.get("device", "?")).replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")


def tables_dir(cfg: TrainConfig) -> Path:
    return cfg.base_dir / "final_results" / "tables"


def results_table(cfg: TrainConfig) -> str:
    """E1-E4: mean ± SD over seeds for every metric."""
    mbc = pd.read_csv(tables_dir(cfg) / "metrics_by_condition.csv")
    names = _names(cfg)
    rows = []
    for i, es in enumerate(CONDITIONS):
        g = mbc[mbc["eval_set"] == es]
        src = {"kvasir_test": "kvasir", "cvc_external": "kvasir", "cvc_test": "cvc", "kvasir_external": "cvc"}[es]
        target = es.rsplit("_", 1)[0]
        cells = []
        for m, d in (("dice", 3), ("iou", 3), ("precision", 3), ("recall", 3), ("hd95", 1), ("boundary_iou", 3), ("miss", 3)):
            r = g[g["metric"] == m].iloc[0]
            cells.append(_pm(r["mean_over_seeds"], r["sd_over_seeds"], d))
        flag = " (weak baseline, E10)" if src == "cvc" else ""
        rows.append([f"E{i + 1}", f"{names[src]} ({_n_train(cfg, src)}){flag}", f"{names[target]} {'test' if es.endswith('_test') else 'whole set'} ({int(g['n_images'].iloc[0])})"] + cells)
    seeds = int(mbc["n_seeds"].max())
    return (_table(rows, ["ID", "Trained on (images)", "Evaluated on (images)", "Dice", "IoU", "Precision", "Recall", "HD95 [px]", "Boundary IoU", "Miss rate"])
            + f"\n\nMean ± sample SD over {seeds} seeds; each seed value is the mean over the images of the evaluation set.")


def gap_table(cfg: TrainConfig) -> str:
    """Generalization gap G = Dice(own test) - Dice(external) with both confidence intervals and the decision rule."""
    gap = pd.read_csv(tables_dir(cfg) / "gap_summary.csv")
    dec = pd.read_csv(tables_dir(cfg) / "decision_rule.csv")
    n_seeds = int(gap["n_seeds"].max())
    rows = []
    for _, r in gap[gap["metric"] == "dice"].iterrows():
        d = dec[dec["direction"] == r["direction"]].set_index("delta")["decision_t_ci"]
        rows.append([r["direction"] + (" (E10)" if isinstance(r["status"], str) and r["status"] else ""), f"{r['own_mean']:.3f}", f"{r['external_mean']:.3f}", f"{r['mean_gap']:.3f}", f"{r['sd_gap']:.3f}",
                     f"[{r['ci_t_low']:.3f}, {r['ci_t_high']:.3f}]", f"[{r['ci_boot_low']:.3f}, {r['ci_boot_high']:.3f}]", d[0.05], d[0.03], d[0.10]])
    return _table(rows, ["Model", "Own test", "External", "Gap G", "SD of G", f"95% CI (t, df = {n_seeds - 1})", "95% CI (cluster bootstrap)", "δ = 0.05", "δ = 0.03", "δ = 0.10"])


def gap_by_seed_table(cfg: TrainConfig) -> str:
    """Dice gap of every seed with the machine that trained it (the machine is confounded with the seed)."""
    t = pd.read_csv(tables_dir(cfg) / "gap_by_seed.csv")
    t = t[t["metric"] == "dice"]
    dirs = list(t["direction"].unique())
    seeds = sorted(t["seed"].unique())
    rows, per_dir = [], {d: [] for d in dirs}
    for seed in seeds:
        cells = []
        for d in dirs:
            g = float(t[(t["direction"] == d) & (t["seed"] == seed)]["gap"].iloc[0])
            cells.append(f"{g:.3f}")
            per_dir[d].append((seed, g))
        rows.append([seed, _gpu(cfg.runs_dir / f"{cfg.sources[0]}_seed{seed}")] + cells)
    rows.append(["mean of seeds " + ", ".join(str(x) for x in seeds if x != seeds[0]), "same machine for all of them"]
                + [f"{np.mean([g for sd, g in per_dir[d] if sd != seeds[0]]):.3f}" for d in dirs])
    return _table(rows, ["Seed", "Trained on"] + dirs)


def per_run_table(cfg: TrainConfig) -> str:
    """Every finished run with its stopping point and its Dice on the validation, own-test and external sets."""
    names = _names(cfg)
    rows = []
    for run in sorted(p for p in cfg.runs_dir.iterdir() if (p / "eval_summary.json").is_file()):
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        ev = json.loads((run / "eval_summary.json").read_text(encoding="utf-8"))["sets"]
        src = meta["source"]
        other = [k for k in names if k != src][0]
        own, ext = ev[f"{src}_test"], ev[f"{other}_external"]
        rows.append([f"{names[src]}", meta["seed"], meta["n_train"], meta["epochs_run"], meta["best_epoch"], f"{meta['best_val_dice']:.3f}",
                     f"{own['mean_dice']:.3f} ({own['n_images']})", f"{ext['mean_dice']:.3f} ({ext['n_images']})"])
    planned = len(cfg.sources) * len(cfg.seeds)
    return (_table(rows, ["Trained on", "Seed", "Train images", "Epochs", "Best epoch", "Val Dice", "Own test Dice (images)", "External Dice (images)"])
            + f"\n\n**Status: {len(rows)} of {planned} planned runs finished.**")


def errors_block(cfg: TrainConfig) -> str:
    """Quantitative failure indicators: poorly segmented frames, empty predictions, size dependence."""
    tdir = tables_dir(cfg)
    mbc = pd.read_csv(tdir / "metrics_by_condition.csv")
    sp = pd.read_csv(tdir / "spearman.csv")
    df = pd.read_csv(cfg.base_dir / "final_results" / "per_image" / "per_image_metrics_extended.csv", dtype={"sequence_id": str})
    b = size_bins(df)
    seed_mean = df[(df["source"] == "kvasir") & (df["eval_set"] == "cvc_external")].groupby("stem")["dice"].mean()
    lines = [f"- Kvasir-SEG-trained model on CVC-ClinicDB (E2): {int((seed_mean < 0.5).sum())} of {len(seed_mean)} frames have seed-averaged Dice < 0.5."]
    miss = {es: mbc[(mbc["eval_set"] == es) & (mbc["metric"] == "miss")].iloc[0] for es in CONDITIONS}
    lines.append("- Share of images with an empty prediction (miss rate): " + ", ".join(f"E{i + 1} {miss[es]['mean_over_seeds'] * 100:.1f}%" for i, es in enumerate(CONDITIONS)) + ".")
    for _, r in sp.iterrows():
        if r["variable"] == "gt_area_fraction":
            lines.append(f"- Spearman ρ between per-image Dice and ground-truth polyp area, {r['trained_on']} on {r['evaluated_on']}: {r['spearman_rho']:.2f} (n = {int(r['n_images'])}; descriptive).")
    rows = []
    for i, es in enumerate(CONDITIONS):
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


def build_blocks() -> Dict[str, str]:
    cfg = load_train_config(ROOT / "configs" / "equal_size.yaml")
    return {"per_run": per_run_table(cfg), "results": results_table(cfg), "gap": gap_table(cfg), "gap_by_seed": gap_by_seed_table(cfg), "errors": errors_block(cfg)}


def apply_blocks(text: str, blocks: Dict[str, str]) -> str:
    """Replace the content between the BEGIN/END markers of each block."""
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)
        if not pattern.search(text):
            raise ValueError(f"README.md has no block '{name}'")
        text = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
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
