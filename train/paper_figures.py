"""Figures for the research paper (PNG, 200 dpi, English labels, black text on white).

``write_all`` builds the agreed figure set from ``per_image_metrics_extended.csv``; conditions are numbered
E1 (first dataset, own test), E2 (first dataset, external), E3 (second dataset, own test), E4 (second dataset, external).
Every figure reads only the per-image file and the audit outputs, so the set can be rebuilt with one command.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .config import TrainConfig  # noqa: E402
from .extended import get_device, load_extended, load_model, out_dir, predict_masks  # noqa: E402
from .extended_figures import COLORS, DPI, _style, figure_gap, figure_training_curves  # noqa: E402
from .extended_stats import (HIGHER_IS_BETTER, METRICS, decide_delta, directions, gap_tables, preliminary,  # noqa: E402
                             seed_means, t_interval)

log = logging.getLogger("train")
PANEL_METRICS = ["dice", "iou", "precision", "recall", "hd95", "boundary_iou"]
TITLES = {"dice": "Dice", "iou": "IoU", "precision": "Precision", "recall": "Recall", "hd95": "HD95 (pixels)",
          "boundary_iou": "Boundary IoU", "miss": "Miss rate"}


@dataclass
class Condition:
    """One of the four evaluation conditions E1-E4."""

    code: str            # E1 ... E4
    source: str          # training dataset key
    eval_set: str
    own: bool
    color: str
    status: str
    label: str           # short label for axes


def conditions(cfg: TrainConfig) -> List[Condition]:
    names = {k: v.name for k, v in cfg.audit.datasets.items()}
    out: List[Condition] = []
    for i, (src, label, own, ext) in enumerate(directions(cfg)):
        for j, (es, is_own) in enumerate(((own, True), (ext, False))):
            target = es.rsplit("_", 1)[0]
            kind = "own test" if is_own else "external"
            out.append(Condition(f"E{2 * i + j + 1}", src, es, is_own, COLORS[i % 2], preliminary(cfg, src),
                                 f"E{2 * i + j + 1}  {names[src]} → {names[target]}\n({kind})" + (f"\n{preliminary(cfg, src)}" if preliminary(cfg, src) else "")))
    return out


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return path


def _stem_values(df: pd.DataFrame, eval_set: str, metric: str) -> pd.Series:
    """Metric averaged over seeds for each image of an evaluation set."""
    return df[df["eval_set"] == eval_set].groupby("stem")[metric].mean()


# ---------------------------------------------------------------- F1 study design
def fig_study_design(cfg: TrainConfig, path: Path) -> Path:
    from .data import load_items
    keys = list(cfg.audit.datasets)
    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 6.5)
    ax.axis("off")
    conds = conditions(cfg)

    def box(x, y, w, h, text, color, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", facecolor="white", edgecolor=color, linewidth=2.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="black")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="black", linewidth=1.2))

    for i, key in enumerate(keys):
        parts, all_items = load_items(cfg, key)
        ds = cfg.audit.datasets[key]
        y = 4.0 - i * 3.0
        box(0.2, y, 3.0, 1.5, f"{ds.name}\n{len(all_items)} images\nsplit: train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])}", COLORS[i % 2])
        box(4.3, y, 3.0, 1.5, f"U-Net trained on\n{ds.name}\n({len(parts['train'])} training images, {len(cfg.seeds)} seeds)", COLORS[i % 2])
        arrow(3.3, y + 0.75, 4.2, y + 0.75)
        own, ext = conds[2 * i], conds[2 * i + 1]
        other = [k for k in keys if k != key][0]
        box(8.4, y + 0.85, 3.4, 0.9, f"{own.code}: own test ({len(parts['test'])} images)", COLORS[i % 2], 9)
        box(8.4, y - 0.15, 3.4, 0.9, f"{ext.code}: whole {cfg.audit.datasets[other].name}\n({len(load_items(cfg, other)[1])} images, external)", COLORS[i % 2], 9)
        arrow(7.4, y + 0.95, 8.3, y + 1.3)
        arrow(7.4, y + 0.55, 8.3, y + 0.3)
    ax.text(0.2, 6.2, "Datasets", fontsize=11, fontweight="bold", color="black")
    ax.text(4.3, 6.2, "Models", fontsize=11, fontweight="bold", color="black")
    ax.text(8.4, 6.2, "Evaluation (Dice per image)", fontsize=11, fontweight="bold", color="black")
    ax.text(0.2, -0.3, "Generalization gap G = Dice(own test) - Dice(external). Validation parts select the best epoch; test and external sets are never used for training or selection.",
            fontsize=8.5, color="black")
    return _save(fig, path)


# ---------------------------------------------------------------- F2 dataset differences
def fig_dataset_differences(cfg: TrainConfig, path: Path) -> Path:
    keys = list(cfg.audit.datasets)
    tables = {k: pd.read_csv(cfg.audit.output_dir / f"per_image_stats_{k}.csv") for k in keys}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="white")
    for i, k in enumerate(keys):
        t, name, col = tables[k], cfg.audit.datasets[k].name, COLORS[i % 2]
        axes[0].hist(t["polyp_area_fraction"] * 100, bins=np.linspace(0, 90, 46), density=True, color=col, alpha=0.4, linewidth=0)
        axes[0].hist(t["polyp_area_fraction"] * 100, bins=np.linspace(0, 90, 46), density=True, histtype="step", color=col, linewidth=2, label=f"{name} (n={len(t)}, median {t['polyp_area_fraction'].median() * 100:.1f}%)")
        axes[1].hist(t["brightness_gray"], bins=np.linspace(0, 160, 41), density=True, color=col, alpha=0.4, linewidth=0)
        axes[1].hist(t["brightness_gray"], bins=np.linspace(0, 160, 41), density=True, histtype="step", color=col, linewidth=2, label=f"{name} (mean {t['brightness_gray'].mean():.1f})")
        sizes = t.groupby(["img_width", "img_height"]).size().reset_index(name="n")
        axes[2].scatter(sizes["img_width"], sizes["img_height"], s=70 if len(sizes) < 4 else 22, color=col, alpha=0.75, marker="o" if i == 0 else "^", edgecolors="white", linewidths=0.8,
                        label=f"{name} ({len(sizes)} distinct size{'s' if len(sizes) != 1 else ''})")
    for ax, (title, xl, yl) in zip(axes, [("Polyp area per image", "Polyp area (% of image)", "Density"), ("Mean image brightness", "Mean grey level (0-255)", "Density"), ("Image size", "Width (px)", "Height (px)")]):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        _style(ax)
        ax.legend(frameon=False, labelcolor="black", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- F3 Dice by condition
def fig_dice_by_condition(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    conds = conditions(cfg)
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor="white")
    for pos, c in enumerate(conds):
        v = _stem_values(df, c.eval_set, "dice").to_numpy()
        bp = ax.boxplot(v, positions=[pos], widths=0.55, patch_artist=True, showfliers=False)
        rgb = matplotlib.colors.to_rgb(c.color)
        for b in bp["boxes"]:
            b.set(facecolor=(*rgb, 0.4) if c.own else "none", edgecolor=c.color, linewidth=2)
        for part in ("whiskers", "caps", "medians"):
            for ln in bp[part]:
                ln.set(color=c.color, linewidth=2)
        ax.scatter(pos + np.random.default_rng(pos).uniform(-0.2, 0.2, len(v)), v, s=5, color=c.color, alpha=0.28, linewidths=0)
        sm = seed_means(df, c.eval_set, "dice")
        ax.scatter(np.full(len(sm), pos + 0.34) + np.linspace(-0.03, 0.03, len(sm)), sm.to_numpy(), s=26, color="black", zorder=5)
        ax.text(pos, 1.04, f"mean {sm.mean():.3f}", ha="center", fontsize=9, color="black")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.label for c in conds], fontsize=8)
    ax.set_ylim(-0.02, 1.1)
    ax.set_ylabel("Dice per image (mean over seeds)")
    ax.set_title("Dice by condition", loc="left", fontweight="bold")
    _style(ax)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markersize=5, label="seed means"),
                        Line2D([0], [0], color="#555555", lw=2, label="filled box: own test; open box: external; dots: images (mean over seeds)")],
               frameon=False, loc="lower center", ncol=2, fontsize=8, labelcolor="black")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path)


# ---------------------------------------------------------------- F5 all metrics
def fig_all_metrics(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    conds = conditions(cfg)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), facecolor="white")
    for ax, metric in zip(axes.ravel(), PANEL_METRICS):
        for pos, c in enumerate(conds):
            sm = seed_means(df, c.eval_set, metric).to_numpy()
            t = t_interval(sm)
            ax.scatter(np.full(len(sm), pos) + np.linspace(-0.12, 0.12, len(sm)), sm, s=16, color=c.color, alpha=0.55, zorder=3)
            ax.errorbar([pos + 0.28], [t["mean"]], yerr=[[t["mean"] - t["low"]], [t["high"] - t["mean"]]], fmt="o" if c.own else "s", color=c.color,
                        markerfacecolor=c.color if c.own else "white", markersize=8, capsize=5, linewidth=2, zorder=4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.code + (" *" if c.status else "") for c in conds], fontsize=9)
        ax.set_title(f"{TITLES[metric]} ({'higher' if HIGHER_IS_BETTER[metric] else 'lower'} is better)", loc="left", fontsize=10, fontweight="bold")
        _style(ax)
    fig.text(0.01, 0.005, "small dots: seeds; large marker: mean with t-based 95% CI over seeds; filled: own test, open: external. "
             + " ".join(f"{c.code}: {c.label.splitlines()[0][len(c.code) + 2:]} ({c.label.splitlines()[1].strip('()')})." for c in conds)
             + (" * preliminary (E10)." if any(c.status for c in conds) else ""), fontsize=7.5, color="black", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path)


# ---------------------------------------------------------------- F6 gap for all metrics
def fig_gap_all_metrics(cfg: TrainConfig, by_seed: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    dirs = directions(cfg)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), facecolor="white")
    for ax, metric in zip(axes.ravel(), METRICS):
        for i, (key, label, own, ext) in enumerate(dirs):
            g = by_seed[(by_seed["direction"] == label) & (by_seed["metric"] == metric)]
            r = summary[(summary["direction"] == label) & (summary["metric"] == metric)].iloc[0]
            ax.scatter(g["gap"], np.full(len(g), i) + np.linspace(-0.08, 0.08, len(g)), s=16, color=COLORS[i % 2], alpha=0.55, zorder=3)
            ax.errorbar([r["mean_gap"]], [i + 0.25], xerr=[[r["mean_gap"] - r["ci_t_low"]], [r["ci_t_high"] - r["mean_gap"]]], fmt="D", color=COLORS[i % 2], markersize=7, capsize=5, linewidth=2, zorder=4)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(range(len(dirs)))
        ax.set_yticklabels([f"{d[1].replace('-trained', '')}" + (" *" if preliminary(cfg, d[0]) else "") for d in dirs], fontsize=8)
        ax.set_ylim(-0.6, len(dirs) - 0.2)
        ax.set_title(f"{TITLES[metric]}  ({'higher' if HIGHER_IS_BETTER[metric] else 'lower'} is better)", loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("gap = own test - external", fontsize=8)
        _style(ax)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.7, "dots: seeds\ndiamond: mean with\nt-based 95% CI over seeds\n" + ("* preliminary (E10)" if any(preliminary(cfg, d[0]) for d in dirs) else ""), fontsize=9, color="black", va="top")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- F7 Dice by polyp size
def size_bins(df: pd.DataFrame) -> Dict[str, float]:
    """Quartiles of the ground-truth area fraction over all distinct evaluated images (pooled over datasets)."""
    area = df.drop_duplicates(["eval_set", "stem"])["gt_area"]
    return {"q1": float(area.quantile(0.25)), "q3": float(area.quantile(0.75))}


def fig_dice_by_size(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    conds = conditions(cfg)
    b = size_bins(df)
    names = ["small", "medium", "large"]
    cut = lambda a: np.where(a < b["q1"], "small", np.where(a > b["q3"], "large", "medium"))  # noqa: E731
    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor="white")
    width = 0.19
    for j, c in enumerate(conds):
        g = df[df["eval_set"] == c.eval_set].copy()
        g["bin"] = cut(g["gt_area"].to_numpy())
        for k, name in enumerate(names):
            sm = g[g["bin"] == name].groupby("seed")["dice"].mean().to_numpy()
            n_img = int(g[g["bin"] == name]["stem"].nunique())
            if len(sm) == 0:
                continue
            t = t_interval(sm) if len(sm) > 1 else {"mean": float(sm[0]), "low": float(sm[0]), "high": float(sm[0])}
            x = k + (j - 1.5) * width
            ax.errorbar([x], [t["mean"]], yerr=[[t["mean"] - t["low"]], [t["high"] - t["mean"]]], fmt="o" if c.own else "s", color=c.color, markerfacecolor=c.color if c.own else "white",
                        markersize=8, capsize=4, linewidth=2)
            ax.text(x, 1.01, f"n={n_img}", ha="center", fontsize=6.5, color="black", rotation=90, va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{n}\n(area {r})" for n, r in zip(names, [f"< {b['q1'] * 100:.1f}%", f"{b['q1'] * 100:.1f}-{b['q3'] * 100:.1f}%", f"> {b['q3'] * 100:.1f}%"])], fontsize=9)
    ax.set_ylim(-0.02, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Dice (mean over seeds, t-based 95% CI)")
    ax.set_title("Dice by polyp size (ground-truth area at 256 px; bins = pooled quartiles)", loc="left", fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="o" if c.own else "s", color=c.color, markerfacecolor=c.color if c.own else "white", markersize=8, lw=2, label=c.label.replace("\n", " ")) for c in conds],
              frameon=False, fontsize=7.5, labelcolor="black", loc="lower right")
    _style(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path)


# ---------------------------------------------------------------- F8 qualitative examples
def fig_examples(cfg: TrainConfig, df: pd.DataFrame, cond: Condition, out: Path, seed: int = 1) -> Path:
    """Best, median and worst frame (by the Dice of the shown model) of an external set: image, ground truth, prediction, overlay."""
    from .data import SegDataset
    from .evaluate import build_eval_sets
    device = get_device(cfg)
    items = next(it for name, _, _, it, _ in build_eval_sets(cfg, cond.source) if name == cond.eval_set)
    d = df[(df["eval_set"] == cond.eval_set) & (df["seed"] == seed) & (df["source"] == cond.source)].sort_values(["dice", "stem"]).reset_index(drop=True)
    picks = [("worst", d.iloc[0]), ("median", d.iloc[len(d) // 2]), ("best", d.iloc[-1])]
    index = {it.stem: i for i, it in enumerate(items)}
    chosen = [items[index[r["stem"]]] for _, r in picks]
    ds = SegDataset(chosen, cfg.image_size, cfg.mask_threshold, None, False)
    model = load_model(cfg.runs_dir / f"{cond.source}_seed{seed}", int(cfg.section("model").get("base_channels", 32)), device)
    preds, _ = predict_masks(model, ds, device, 8, float(cfg.section("training").get("threshold", 0.5)))
    green, orange = np.array([0.10, 0.65, 0.30]), np.array([0.92, 0.41, 0.20])
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6), facecolor="white")
    for r, ((label, row), pred) in enumerate(zip(picks, preds)):
        img, gt = ds.images[r] / 255.0, ds.masks[r].astype(bool)
        overlay = img.copy()
        overlay[pred] = 0.55 * overlay[pred] + 0.45 * orange
        panels = [img, gt, pred, overlay]
        for k, panel in enumerate(panels):
            ax = axes[r][k]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1) if panel.ndim == 2 else ax.imshow(panel)
            if k == 3:
                ax.contour(gt.astype(float), levels=[0.5], colors=[green], linewidths=1.2)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(f"{label}\nDice {row['dice']:.2f}", fontsize=10, color="black")
    for k, t in enumerate(["Image", "Ground truth", f"Prediction (seed {seed})", "Overlay"]):
        axes[0][k].set_title(t, fontsize=9, color="black")
    fig.suptitle(f"{cond.code}: " + cond.label.splitlines()[0][len(cond.code) + 2:] + " (external set)" + (f"  [{cond.status}]" if cond.status else ""), x=0.01, ha="left", fontsize=11, fontweight="bold", color="black")
    fig.text(0.01, 0.005, "Overlay: prediction in orange, ground-truth outline in green.", fontsize=8.5, color="black")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, out)


# ---------------------------------------------------------------- F9 equal-size comparison
def fig_equal_size(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Optional[Path]:
    """Dice of Kvasir-800 (first experiment), Kvasir-429 and CVC-429 (this experiment) on own test and external sets."""
    e = cfg.section("extended")
    old_csv = (cfg.base_dir / e.get("compare_to", "results/extended/per_image_metrics_extended.csv")).resolve()
    old_runs = (cfg.base_dir / e.get("compare_runs", "runs")).resolve()
    if not old_csv.is_file() or old_runs == cfg.runs_dir:
        log.info("equal-size comparison skipped: no earlier experiment found at %s", old_csv)
        return None
    old = pd.read_csv(old_csv, dtype={"sequence_id": str})
    names = {k: v.name for k, v in cfg.audit.datasets.items()}

    def n_train(runs: Path, src: str) -> int:
        return int(json.loads(next(runs.glob(f"{src}_seed*/run.json")).read_text(encoding="utf-8"))["n_train"])

    arms = []
    for i, (src, label, own, ext) in enumerate(directions(cfg)):
        if (old["source"] == src).any() and n_train(old_runs, src) != n_train(cfg.runs_dir, src):
            arms.append((f"{names[src]}, {n_train(old_runs, src)} images\n(first experiment)", old, src, own, ext, COLORS[i % 2], "^"))
        arms.append((f"{names[src]}, {n_train(cfg.runs_dir, src)} images\n(equal-size experiment)", df, src, own, ext, COLORS[i % 2], "o"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), facecolor="white", sharey=True)
    for ax, (title, which) in zip(axes, [("Own test set", 2), ("External set (the other dataset)", 3)]):
        for x, arm in enumerate(arms):
            label, data, src, own, ext, color, marker = arm
            es = own if which == 2 else ext
            sm = data[(data["source"] == src) & (data["eval_set"] == es)].groupby("seed")["dice"].mean().to_numpy()
            t = t_interval(sm)
            ax.scatter(np.full(len(sm), x) + np.linspace(-0.12, 0.12, len(sm)), sm, s=16, color=color, alpha=0.55, zorder=3, marker=marker)
            ax.errorbar([x + 0.25], [t["mean"]], yerr=[[t["mean"] - t["low"]], [t["high"] - t["mean"]]], fmt=marker, color=color, markersize=8, capsize=5, linewidth=2, zorder=4)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], fontsize=7.5)
        ax.set_title(title, loc="left", fontweight="bold")
        _style(ax)
    axes[0].set_ylabel("Dice (mean over seeds, t-based 95% CI)")
    fig.text(0.01, 0.005, "dots: seeds; marker with bar: mean and t-based 95% CI over seeds. Own test sets differ between datasets (Kvasir-SEG 100, CVC-ClinicDB 91 images).", fontsize=8, color="black")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, path)


# ---------------------------------------------------------------- appendix
def fig_delta_sensitivity(cfg: TrainConfig, summary: pd.DataFrame, path: Path) -> Path:
    deltas = [float(x) for x in cfg.section("analysis").get("practical_thresholds", [0.05, 0.03, 0.10])]
    dice = summary[summary["metric"] == "dice"].reset_index(drop=True)
    fig, axes = plt.subplots(1, len(deltas), figsize=(5.2 * len(deltas), 3.6), facecolor="white", sharex=True)
    for ax, delta in zip(np.atleast_1d(axes), deltas):
        for i, r in dice.iterrows():
            ax.errorbar([r["mean_gap"]], [i], xerr=[[r["mean_gap"] - r["ci_t_low"]], [r["ci_t_high"] - r["mean_gap"]]], fmt="D", color=COLORS[i % 2], markersize=7, capsize=6, linewidth=2)
            ax.text(r["ci_t_high"] + 0.01, i, decide_delta(r["ci_t_low"], r["ci_t_high"], delta), va="center", fontsize=8, color="black")
        for s in (-delta, delta):
            ax.axvline(s, color="#555555", linestyle=":", linewidth=1.3)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(range(len(dice)))
        ax.set_yticklabels([r["direction"].replace("-trained", "") + (" *" if isinstance(r["status"], str) and r["status"] else "") for _, r in dice.iterrows()], fontsize=8)
        ax.set_title(f"delta = {delta:g}", loc="left", fontweight="bold")
        ax.set_xlabel("Dice gap (own test - external), t-based 95% CI")
        ax.set_ylim(-0.6, len(dice) - 0.4)
        _style(ax)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_xlim(min(-0.12, dice["ci_t_low"].min() - 0.05), max(0.45, dice["ci_t_high"].max() + 0.25))
    fig.tight_layout()
    return _save(fig, path)


def fig_error_categories(cfg: TrainConfig, path: Path) -> Optional[Path]:
    from .extended import error_dir
    f = error_dir(cfg) / "category_counts.csv"
    if not f.is_file():
        return None
    c = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="white")
    x = np.arange(len(c))
    ax.bar(x - 0.2, c["annotator_A"], 0.38, color=COLORS[0], label="annotator A")
    ax.bar(x + 0.2, c["annotator_B"], 0.38, color=COLORS[1], label="annotator B")
    ax.set_xticks(x)
    ax.set_xticklabels(c["category"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Frames")
    ax.set_title("Error categories (main direction)", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor="black")
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- driver
def write_all(cfg: TrainConfig, target: Path, with_examples: bool = True, with_comparison: bool = True) -> List[Path]:
    """Build the whole figure set into ``target``; returns the files written.

    ``with_comparison`` adds the comparison with an earlier experiment (``extended.compare_to``) when such a file exists.
    """
    target.mkdir(parents=True, exist_ok=True)
    df = load_extended(cfg)
    by_seed, summary = gap_tables(df, cfg)
    conds = conditions(cfg)
    written: List[Path] = [
        fig_study_design(cfg, target / "fig01_study_design.png"),
        fig_dataset_differences(cfg, target / "fig02_dataset_differences.png"),
        fig_dice_by_condition(cfg, df, target / "fig03_dice_by_condition.png"),
    ]
    figure_gap(cfg, by_seed, summary, target / "fig04_gap_per_seed_dice.png")
    written.append(target / "fig04_gap_per_seed_dice.png")
    written += [fig_all_metrics(cfg, df, target / "fig05_all_metrics.png"),
                fig_gap_all_metrics(cfg, by_seed, summary, target / "fig06_gap_all_metrics.png"),
                fig_dice_by_size(cfg, df, target / "fig07_dice_by_polyp_size.png")]
    if with_examples:
        for c in conds:
            if not c.own:
                written.append(fig_examples(cfg, df, c, target / f"fig08_examples_{c.code}.png"))
    eq = fig_equal_size(cfg, df, target / "fig09_equal_size_comparison.png") if with_comparison else None
    if eq:
        written.append(eq)
    figure_training_curves(cfg, target / "figA1_training_curves.png")
    written.append(target / "figA1_training_curves.png")
    written.append(fig_delta_sensitivity(cfg, summary, target / "figA2_delta_sensitivity.png"))
    err = fig_error_categories(cfg, target / "figA3_error_categories.png")
    if err:
        written.append(err)
    return written
