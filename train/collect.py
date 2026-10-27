"""Collect everything of an experiment in one folder: ``python -m train.collect --config configs/equal_size.yaml``.

Runs the whole analysis chain (statistics, per-image export, extended metrics, figures) and copies the results,
training logs, configuration, splits and a MANIFEST with checksums into one output folder (default ``final_results/``).
Weights are not copied unless ``--with-weights`` is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import analyze as analysis
from . import export, extended, extended_figures, extended_report, extended_stats, paper_figures
from .config import TrainConfig, load_train_config
from .extended import EXTENDED_FILE, out_dir as extended_dir

log = logging.getLogger("train")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def run_analysis_chain(cfg: TrainConfig, skip_inference: bool = False, with_errors: bool = False, progress: bool = True) -> None:
    """All analysis stages in order; inference is skipped if requested and its result already exists."""
    analysis.analyze(cfg)
    _export(cfg)
    ext_file = extended_dir(cfg) / EXTENDED_FILE
    if skip_inference and ext_file.is_file():
        log.info("using the existing %s", EXTENDED_FILE)
    else:
        extended.stage_inference(cfg, progress=progress)
    extended_stats.run(cfg)
    extended_figures.run(cfg)
    if with_errors:
        from . import error_material
        error_material.run(cfg)
    extended_report.run(cfg)


def _export(cfg: TrainConfig) -> None:
    df = export.build_per_image_metrics(cfg)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "per_image_metrics.csv", index=False, float_format="%.6f")


def gather(cfg: TrainConfig, config_file: Path, out: Path, with_weights: bool = False, with_examples: bool = False) -> List[Path]:
    """Copy the results into ``out`` and build the paper figures there. Returns all files written.

    Figures that show frames of the datasets (qualitative examples) are skipped unless ``with_examples`` is set:
    the dataset terms restrict redistribution (CVC-ClinicDB: research and education only), so they are kept out of the repository.
    """
    if out.exists():
        shutil.rmtree(out)                      # the folder is rebuilt from scratch so that it never mixes two states
    written: List[Path] = []
    ext = extended_dir(cfg)

    # tables
    for f in sorted((cfg.results_dir / "tables").glob("*.csv")):
        written.append(_copy(f, out / "tables" / "original_analysis" / f.name))
    for name in ("metrics_by_condition.csv", "gap_by_seed.csv", "gap_summary.csv", "decision_rule.csv", "spearman.csv", "acceptance_checks.json"):
        written.append(_copy(ext / name, out / "tables" / name))
    written.append(_copy(cfg.results_dir / "summary.md", out / "tables" / "original_analysis" / "summary.md"))
    written.append(_copy(ext / "METRICS.md", out / "METRICS.md"))
    written.append(_copy(ext / "README.md", out / "tables" / "extended_report.md"))

    # per-image data
    written.append(_copy(ext / EXTENDED_FILE, out / "per_image" / EXTENDED_FILE))
    written.append(_copy(cfg.results_dir / "per_image_metrics.csv", out / "per_image" / "per_image_metrics.csv"))

    # figures
    written += paper_figures.write_all(cfg, out / "figures", with_examples=with_examples)
    dist = ext / "figures" / "fig_c_metric_distributions.png"
    if dist.is_file():
        written.append(_copy(dist, out / "figures" / "fig10_metric_distributions.png"))

    # training logs of every run
    for run in sorted(p for p in cfg.runs_dir.iterdir() if (p / "run.json").is_file()):
        for name in ("metrics.csv", "run.json", "eval_summary.json", "per_image_dice.csv"):
            if (run / name).is_file():
                written.append(_copy(run / name, out / "runs" / run.name / name))
        if with_examples:
            for png in sorted(run.glob("examples_*.png")):
                written.append(_copy(png, out / "runs" / run.name / png.name))
        if with_weights and (run / "best.pt").is_file():
            written.append(_copy(run / "best.pt", out / "weights" / f"{run.name}.pt"))
    for log_file in sorted(cfg.runs_dir.rglob("*.log")):
        written.append(_copy(log_file, out / "runs" / log_file.relative_to(cfg.runs_dir)))

    # configuration, splits and audit
    for f in (config_file, cfg.base_dir / "train_config.yaml", cfg.base_dir / "config.yaml"):
        if f.is_file():
            written.append(_copy(f, out / "config" / f.name))
    for f in sorted(cfg.splits_dir.glob("*")):
        if f.is_file():
            written.append(_copy(f, out / "config" / "splits" / f.name))
    report = cfg.audit.output_dir / "audit_report.md"
    if report.is_file():
        written.append(_copy(report, out / "audit" / "audit_report.md"))
    return written


def key_numbers(cfg: TrainConfig) -> str:
    """Markdown tables for the README of the folder, read from the collected statistics."""
    mbc = pd.read_csv(extended_dir(cfg) / "metrics_by_condition.csv")
    gap = pd.read_csv(extended_dir(cfg) / "gap_summary.csv")
    dec = pd.read_csv(extended_dir(cfg) / "decision_rule.csv")
    conds = paper_figures.conditions(cfg)
    rows = []
    for c in conds:
        r = mbc[(mbc["eval_set"] == c.eval_set) & (mbc["metric"] == "dice")].iloc[0]
        rows.append({"condition": c.label.replace("\n", " "), "images": int(r["n_images"]), "seeds": int(r["n_seeds"]), "Dice mean": f"{r['mean_over_seeds']:.3f}", "SD over seeds": f"{r['sd_over_seeds']:.3f}"})
    g = gap[gap["metric"] == "dice"][["direction", "mean_gap", "ci_t_low", "ci_t_high", "ci_boot_low", "ci_boot_high", "status"]].fillna("")
    d = dec[dec["main_delta"]][["direction", "delta", "decision_t_ci", "decision_bootstrap_ci"]]
    return "\n\n".join([extended_report.md_table(pd.DataFrame(rows)), "Dice gap (own test minus external), t-based CI (primary) and cluster bootstrap CI:\n\n" + extended_report.md_table(g),
                        "Decision rule, main threshold:\n\n" + extended_report.md_table(d)])


def write_readme_and_manifest(cfg: TrainConfig, out: Path, config_file: Path, files: List[Path]) -> None:
    runs = sorted(p.name for p in cfg.runs_dir.iterdir() if (p / "run.json").is_file())
    text = f"""# Collected results

Everything in this folder was produced from `{config_file.name}` by `python -m train.collect`. Nothing here is edited by hand.

- Runs: {len(runs)} ({', '.join(runs)}).
- Models: U-Net, {cfg.section('model').get('base_channels', 32)} base channels, 256 x 256 input; metrics are computed on the 256 x 256 masks at probability threshold {cfg.section('training').get('threshold', 0.5)}.

## Contents
| Folder or file | Content |
|---|---|
| `figures/` | figures for the paper: fig01 study design, fig02 dataset differences, fig03 Dice by condition, fig04 gap per seed, fig05 all metrics, fig06 gap for all metrics, fig07 Dice by polyp size, fig08 qualitative examples, fig09 equal-size comparison (if an earlier experiment exists), fig10 metric distributions, figA* appendix |
| `tables/` | `metrics_by_condition.csv` (mean and SD over seeds, every metric), `gap_by_seed.csv`, `gap_summary.csv` (t-based and bootstrap CI), `decision_rule.csv`, `spearman.csv`, `acceptance_checks.json`, `original_analysis/` |
| `per_image/` | one row per run, evaluation set and image with Dice, IoU, precision, recall, HD95, Boundary IoU, miss |
| `runs/` | per run: training log per epoch (`metrics.csv`), settings and environment (`run.json`), evaluation summary, per-image Dice, example figures |
| `config/` | the configuration files used, the fixed splits and the fixed training subset |
| `audit/` | dataset audit report |
| `METRICS.md` | definitions of all metrics and statistics |
| `MANIFEST.json` | list of all files with SHA-256 checksums, versions and the git commit |

## Key numbers
{key_numbers(cfg)}

Dice gaps are reported with the t-based 95% CI over seeds as primary interval. An effect measured here is the combined effect of everything that differs between the datasets; it is not attributed to one factor.
"""
    (out / "README.md").write_text(text, encoding="utf-8")
    info: Dict[str, object] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "platform": platform.platform()}
    try:
        import subprocess
        root = Path(__file__).resolve().parents[1]
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        info["git_commit"], info["git_dirty"] = None, None
    listing = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = {"config": config_file.name, "runs": runs, "environment": info, "n_files": len(listing),
                "files": {p.relative_to(out).as_posix(): {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in listing}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect(cfg: TrainConfig, config_file: Path, out: Path, skip_inference: bool = False, with_weights: bool = False,
            with_errors: bool = False, with_examples: bool = False, progress: bool = True) -> Path:
    """Run the analysis chain and build the collected folder. Returns the folder."""
    run_analysis_chain(cfg, skip_inference, with_errors, progress)
    files = gather(cfg, config_file, out, with_weights, with_examples)
    write_readme_and_manifest(cfg, out, config_file, files)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the analysis and collect all results of an experiment in one folder.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--out", default="final_results", help="output folder, relative to the project folder (it is deleted and rebuilt)")
    ap.add_argument("--skip-inference", action="store_true", help="reuse an existing per_image_metrics_extended.csv")
    ap.add_argument("--with-weights", action="store_true", help="also copy best.pt of every run")
    ap.add_argument("--with-errors", action="store_true", help="also create the error-analysis material")
    ap.add_argument("--with-frames", action="store_true", help="also build and copy figures that show dataset frames (qualitative examples); keep them out of public repositories")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="force the device (e.g. cuda: stop with an error instead of silently using the CPU)")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    config_file = Path(args.config).resolve()
    cfg = load_train_config(config_file)
    if args.device:
        cfg = cfg.override(**{"training.device": args.device, "extended.device": args.device})
    out = collect(cfg, config_file, (cfg.base_dir / args.out).resolve(), args.skip_inference, args.with_weights, args.with_errors, args.with_frames, not args.no_progress)
    log.info("collected results in %s", out.relative_to(cfg.base_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
