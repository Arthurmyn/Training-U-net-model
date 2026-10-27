"""One folder with every figure of the experiment: ``python -m train.figure_gallery --config configs/equal_size.yaml``.

Builds the paper figure set, the additional figures of ``train.extra_figures`` and the metric distributions, sorts them
into topic folders (data, training, results, errors, robustness) and writes an index ``README.md``. The folder is
rebuilt from scratch every time. Needs ``per_image_metrics_extended.csv`` (``python -m train.extended inference``).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from . import extra_figures as xf
from . import extended_figures, paper_figures
from .config import TrainConfig, load_train_config
from .extended import load_extended
from .extended_stats import gap_tables

log = logging.getLogger("train")

# code, topic folder, file name, title, what it shows, data source
CATALOGUE: List[Tuple[str, str, str, str, str, str]] = [
    ("D1", "01_data", "D1_study_design.png", "Study design", "datasets, splits, models and the four conditions E1-E4", "splits, run.json"),
    ("D2", "01_data", "D2_dataset_differences.png", "Dataset differences", "polyp area, mean brightness and image size of both datasets", "outputs/per_image_stats_*.csv"),
    ("D3", "01_data", "D3_colour_polyps_aspect.png", "Colour, polyps per mask, aspect ratio", "colour channels, number of polyps per mask and image aspect ratio", "outputs/per_image_stats_*.csv"),
    ("D4", "01_data", "D4_split_overview.png", "Splits", "train/val/test sizes and frames per video sequence coloured by part", "outputs/splits/*.csv"),
    ("D5", "01_data", "D5_near_duplicates.png", "Near-duplicate candidates", "image pairs per perceptual-hash distance, within and between the datasets", "outputs/duplicate_pairs.csv"),
    ("T1", "02_training", "T1_training_curves.png", "Training curves", "loss and Dice per epoch, one line per seed", "runs_equal/*/metrics.csv"),
    ("T2", "02_training", "T2_validation_and_stopping.png", "Validation Dice and stopping", "validation Dice of every run, selected epoch and stopping epoch", "runs_equal/*/metrics.csv, run.json"),
    ("T3", "02_training", "T3_train_validation_gap.png", "Train vs validation gap", "train Dice minus validation Dice over the epochs", "runs_equal/*/metrics.csv"),
    ("R1", "03_results", "R1_dice_by_condition.png", "Dice by condition", "per-image Dice distribution and seed means for E1-E4", "per_image_metrics_extended.csv"),
    ("R2", "03_results", "R2_gap_per_seed.png", "Generalization gap per seed", "gap G per seed with the t-based 95% interval and the delta thresholds", "per_image_metrics_extended.csv"),
    ("R3", "03_results", "R3_all_metrics.png", "All metrics", "Dice, IoU, precision, recall, HD95, Boundary IoU per condition (seeds and 95% interval)", "per_image_metrics_extended.csv"),
    ("R4", "03_results", "R4_gap_all_metrics.png", "Gap for all metrics", "gap of every metric with its interval", "per_image_metrics_extended.csv"),
    ("R5", "03_results", "R5_metric_distributions.png", "Metric distributions", "per-image Dice, IoU, HD95 and Boundary IoU", "per_image_metrics_extended.csv"),
    ("R6", "03_results", "R6_delta_sensitivity.png", "Sensitivity to the threshold", "decision of the rule for delta = 0.05, 0.03, 0.10", "gap_summary.csv"),
    ("R7", "03_results", "R7_dice_ecdf.png", "Cumulative distribution of Dice", "share of images below each Dice value for E1-E4", "per_image_metrics_extended.csv"),
    ("R8", "03_results", "R8_paired_same_images.png", "Same images, two models", "the same test images segmented by both models (paired)", "per_image_metrics_extended.csv"),
    ("R9", "03_results", "R9_precision_recall.png", "Precision vs recall", "over- vs under-segmentation per condition and seed", "per_image_metrics_extended.csv"),
    ("F1", "04_errors", "F1_dice_by_polyp_size.png", "Dice by polyp size", "small / medium / large polyps (quartiles of the true area)", "per_image_metrics_extended.csv"),
    ("F2", "04_errors", "F2_dice_vs_polyp_size.png", "Dice vs polyp size (per image)", "continuous dependence of Dice on the true polyp area", "per_image_metrics_extended.csv"),
    ("F3", "04_errors", "F3_predicted_vs_true_area.png", "Predicted vs true area", "calibration of the predicted polyp area; empty predictions", "per_image_metrics_extended.csv"),
    ("F4", "04_errors", "F4_dice_vs_brightness.png", "Dice vs brightness", "descriptive look at the brightness shift (Spearman rho)", "per_image_metrics_extended.csv, outputs/per_image_stats_*.csv"),
    ("F5", "04_errors", "F5_dice_by_sequence.png", "Dice per video sequence", "dependence of the result on the evaluated sequences", "per_image_metrics_extended.csv"),
    ("F6", "04_errors", "F6_examples_E2.png", "Examples E2", "best, median, worst frame of the external set (shows dataset frames)", "weights, data"),
    ("F7", "04_errors", "F7_examples_E4.png", "Examples E4", "best, median, worst frame of the external set (shows dataset frames)", "weights, data"),
    ("B1", "05_robustness", "B1_original_vs_256.png", "Original resolution vs 256 x 256", "per-image Dice at the original size against the 256 x 256 Dice", "results_equal/original_resolution"),
    ("B2", "05_robustness", "B2_gap_by_seed_and_machine.png", "Gap per seed and machine", "gap of each seed with the GPU that trained it", "gap_by_seed.csv, run.json"),
]
ROOT_OF = {c[0]: c for c in CATALOGUE}


def _place(src: Path, out: Path, code: str, built: List[str]) -> None:
    _, topic, name, *_ = ROOT_OF[code]
    dst = out / topic / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dst)
    built.append(code)


def build(cfg: TrainConfig, out: Path, with_examples: bool = True) -> List[str]:
    """Build every figure into ``out``; returns the catalogue codes that were produced."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = out / "_work"
    work.mkdir()
    df = load_extended(cfg)
    by_seed, _ = gap_tables(df, cfg)
    built: List[str] = []

    # figures of the paper set (names fixed by train.paper_figures)
    paper_figures.write_all(cfg, work, with_examples=with_examples, with_comparison=False)
    for code, name in (("D1", "fig01_study_design.png"), ("D2", "fig02_dataset_differences.png"), ("R1", "fig03_dice_by_condition.png"), ("R2", "fig04_gap_per_seed_dice.png"),
                       ("R3", "fig05_all_metrics.png"), ("R4", "fig06_gap_all_metrics.png"), ("F1", "fig07_dice_by_polyp_size.png"), ("F6", "fig08_examples_E2.png"),
                       ("F7", "fig08_examples_E4.png"), ("T1", "figA1_training_curves.png"), ("R6", "figA2_delta_sensitivity.png")):
        if (work / name).is_file():
            _place(work / name, out, code, built)
    if (work / "figA3_error_categories.png").is_file():                       # only if the annotation was done
        dst = out / "04_errors" / "F8_error_categories.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "figA3_error_categories.png"), dst)

    # distributions of the extended metrics
    p = work / "dist.png"
    extended_figures.figure_distributions(cfg, df, p)
    _place(p, out, "R5", built)

    # additional figures; each one may return None when its input is missing
    makers: Dict[str, Callable[[Path], object]] = {
        "D3": lambda q: xf.fig_color_and_polyps(cfg, q), "D4": lambda q: xf.fig_split_overview(cfg, q), "D5": lambda q: xf.fig_near_duplicates(cfg, q),
        "T2": lambda q: xf.fig_val_curves_and_stopping(cfg, q), "T3": lambda q: xf.fig_train_val_gap(cfg, q),
        "R7": lambda q: xf.fig_dice_ecdf(cfg, df, q), "R8": lambda q: xf.fig_paired_same_images(cfg, df, q), "R9": lambda q: xf.fig_precision_recall(cfg, df, q),
        "F2": lambda q: xf.fig_dice_vs_size(cfg, df, q), "F3": lambda q: xf.fig_pred_vs_gt_area(cfg, df, q), "F4": lambda q: xf.fig_dice_vs_brightness(cfg, df, q),
        "F5": lambda q: xf.fig_dice_by_sequence(cfg, df, q), "B1": lambda q: xf.fig_original_vs_256(cfg, q), "B2": lambda q: xf.fig_gap_by_seed_machine(cfg, by_seed, q),
    }
    for code, make in makers.items():
        q = work / f"{code}.png"
        if make(q) is not None and q.is_file():
            _place(q, out, code, built)
        else:
            log.info("figure %s skipped: input not available", code)
    shutil.rmtree(work)
    return built


def write_index(out: Path, built: List[str]) -> None:
    """Index of the folder: one table per topic, only for figures that exist."""
    titles = {"01_data": "Data and design", "02_training": "Training", "03_results": "Results", "04_errors": "Errors and their causes", "05_robustness": "Robustness checks"}
    lines = ["# All figures of the experiment", "",
             "Built by `python -m train.figure_gallery`; the folder is rebuilt from scratch every time. Conditions: E1 = first dataset on its own test set, E2 = first dataset on the other dataset, "
             "E3 = second dataset on its own test set, E4 = second dataset on the first dataset. "
             "Figures that show frames of the datasets (E6, E7) are for research and educational use only.", ""]
    for topic, title in titles.items():
        rows = [c for c in CATALOGUE if c[1] == topic and c[0] in built]
        extra = sorted((out / topic).glob("F8_*.png")) if topic == "04_errors" else []
        if not rows and not extra:
            continue
        lines += [f"## {title}", "", "| File | Figure | What it shows | Data |", "|---|---|---|---|"]
        lines += [f"| [`{c[2]}`]({topic}/{c[2]}) | {c[3]} | {c[4]} | {c[5]} |" for c in rows]
        lines += [f"| [`{p.name}`]({topic}/{p.name}) | Error categories | counts of the annotated failure categories | error_analysis*/category_counts.csv |" for p in extra]
        lines.append("")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build every figure of the experiment into one folder.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--out", default="all_figures", help="output folder relative to the project folder (deleted and rebuilt)")
    ap.add_argument("--no-examples", action="store_true", help="skip the figures that need the weights and show dataset frames")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="device for the example figures (extended.device)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_train_config(args.config)
    if args.device:
        cfg = cfg.override(**{"extended.device": args.device})
    out = (cfg.base_dir / args.out).resolve()
    built = build(cfg, out, with_examples=not args.no_examples)
    write_index(out, built)
    log.info("%d figures in %s", len(list(out.rglob("*.png"))), out.relative_to(cfg.base_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
