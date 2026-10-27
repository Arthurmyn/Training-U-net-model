"""Additional figures built from the per-image metrics, training logs, audit tables and splits (PNG, 200 dpi, white background).

Each function takes ``(cfg, ...)`` and a path; it returns the path, or ``None`` when its input is not available
(for example sequence figures for a dataset without sequence IDs). They are collected by ``train.figure_gallery``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from audit.load import natural_key  # noqa: E402

from .config import TrainConfig  # noqa: E402
from .extended_figures import COLORS, DPI, _shade, _style  # noqa: E402
from .extended_stats import directions  # noqa: E402
from .paper_figures import Condition, _save, conditions  # noqa: E402

log = logging.getLogger("train")
SPLIT_COLORS = {"train": "#555555", "val": "#1baf7a", "test": "#eda100"}      # neutral / aqua / yellow (validated slots 3-4)


def _names(cfg: TrainConfig) -> Dict[str, str]:
    return {k: v.name for k, v in cfg.audit.datasets.items()}


def _audit_table(cfg: TrainConfig, key: str) -> Optional[pd.DataFrame]:
    p = cfg.audit.output_dir / f"per_image_stats_{key}.csv"
    return pd.read_csv(p, dtype={"sequence_id": str}) if p.is_file() else None


def _seed_mean(df: pd.DataFrame, eval_set: str, cols: List[str]) -> pd.DataFrame:
    """Columns averaged over seeds for every image of an evaluation set (indexed by stem)."""
    g = df[df["eval_set"] == eval_set]
    out = g.groupby("stem")[cols].mean()
    for extra in ("image_file", "sequence_id", "target_split"):
        out[extra] = g.groupby("stem")[extra].first()
    return out


# ------------------------------------------------------------------------------------------------ data
def fig_color_and_polyps(cfg: TrainConfig, path: Path) -> Optional[Path]:
    """Colour channels, number of polyps per mask and aspect ratio of the two datasets."""
    keys, names = list(cfg.audit.datasets), _names(cfg)
    tables = {k: _audit_table(cfg, k) for k in keys}
    if any(t is None for t in tables.values()):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
    width = 0.34
    for i, k in enumerate(keys):
        t = tables[k]
        for j, ch in enumerate(("mean_r", "mean_g", "mean_b")):
            bp = axes[0].boxplot(t[ch], positions=[j + (i - 0.5) * width * 1.1], widths=width, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=(*matplotlib.colors.to_rgb(COLORS[i % 2]), 0.45), edgecolor=COLORS[i % 2], linewidth=1.8)
            for part in ("whiskers", "caps", "medians"):
                for ln in bp[part]:
                    ln.set(color=COLORS[i % 2], linewidth=1.8)
        share = pd.cut(t["n_polyps"], [0, 1, 2, 3, 10 ** 6], labels=["1", "2", "3", "4+"]).value_counts(normalize=True).sort_index() * 100
        axes[1].bar(np.arange(4) + (i - 0.5) * 0.38, share.to_numpy(), 0.36, color=COLORS[i % 2], label=names[k])
        ratio = (t["img_width"] / t["img_height"]).to_numpy()
        axes[2].hist(ratio, bins=np.linspace(0.8, 1.6, 33), weights=np.full(len(ratio), 100.0 / len(ratio)), color=COLORS[i % 2], alpha=0.5, label=f"{names[k]} (median {np.median(ratio):.2f})")
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["red", "green", "blue"])
    axes[0].set_ylabel("Mean channel value per image (0-255)")
    axes[0].set_title("Colour channels", loc="left", fontweight="bold")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["1", "2", "3", "4+"])
    axes[1].set_xlabel("Connected polyp regions in the mask")
    axes[1].set_ylabel("Share of images (%)")
    axes[1].set_title("Polyps per mask", loc="left", fontweight="bold")
    axes[2].set_xlabel("Width / height of the original image")
    axes[2].set_ylabel("Share of images (%)")
    axes[2].set_title("Aspect ratio (resized to 1:1 for the model)", loc="left", fontweight="bold")
    for ax in axes:
        _style(ax)
    axes[1].legend(frameon=False, labelcolor="black", fontsize=8)
    axes[2].legend(frameon=False, labelcolor="black", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def fig_split_overview(cfg: TrainConfig, path: Path) -> Optional[Path]:
    """Sizes of the train / val / test parts and, where sequence IDs exist, the frames per sequence coloured by part."""
    names = _names(cfg)
    splits = {}
    for k in cfg.audit.datasets:
        f = cfg.splits_dir / f"{k}_split.csv"
        if not f.is_file():
            return None
        splits[k] = pd.read_csv(f, dtype={"sequence_id": str})
    seq_keys = [k for k, s in splits.items() if "sequence_id" in s.columns and s["sequence_id"].notna().any()]
    fig, axes = plt.subplots(1, 1 + len(seq_keys), figsize=(6 + 7 * len(seq_keys), 4.6), facecolor="white", squeeze=False)
    ax = axes[0][0]
    for i, (k, s) in enumerate(splits.items()):
        left = 0
        for part in ("train", "val", "test"):
            n = int((s["split"] == part).sum())
            ax.barh(i, n, left=left, color=SPLIT_COLORS[part], edgecolor="white", label=part if i == 0 else None)
            ax.text(left + n / 2, i, str(n), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += n
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([names[k] for k in splits])
    ax.set_xlabel("Images")
    ax.set_title("Fixed splits", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor="black", ncol=1, loc="upper right", fontsize=9)
    _style(ax)
    for j, k in enumerate(seq_keys):
        s = splits[k]
        grp = s.groupby("sequence_id").agg(n=("split", "size"), part=("split", "first")).reset_index()
        grp = grp.sort_values("sequence_id", key=lambda c: c.map(natural_key)).reset_index(drop=True)
        axj = axes[0][1 + j]
        axj.bar(grp.index, grp["n"], color=[SPLIT_COLORS[p] for p in grp["part"]], edgecolor="white")
        axj.set_xticks(grp.index)
        axj.set_xticklabels(grp["sequence_id"], fontsize=7)
        axj.set_xlabel("Video sequence")
        axj.set_ylabel("Frames")
        axj.set_title(f"{names[k]}: frames per video sequence", loc="left", fontweight="bold")
        _style(axj)
    fig.tight_layout()
    return _save(fig, path)


def fig_near_duplicates(cfg: TrainConfig, path: Path) -> Optional[Path]:
    """Number of image pairs at each perceptual-hash distance (0-8 bits), within and between the datasets."""
    f = cfg.audit.output_dir / "duplicate_pairs.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    scopes = sorted(d["scope"].unique())
    names = _names(cfg)
    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor="white")
    palette = {"within_kvasir": COLORS[0], "within_cvc": COLORS[1], "across_datasets": "#555555"}
    for i, sc in enumerate(scopes):
        counts = d[d["scope"] == sc]["hamming"].value_counts().reindex(range(0, 9), fill_value=0)
        label = sc.replace("within_", "within ").replace("across_datasets", "between datasets")
        for key, name in names.items():
            label = label.replace(key, name)
        ax.bar(np.arange(9) + (i - (len(scopes) - 1) / 2) * 0.26, counts.to_numpy(), 0.25, color=palette.get(sc, "#999999"), label=label)
    thr = cfg.audit.near_duplicate_hamming
    ax.axvline(thr + 0.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(thr + 0.6, ax.get_ylim()[1] * 0.9, f"near-duplicate if distance <= {thr}", fontsize=8, color="black")
    ax.set_yscale("log")
    ax.set_xticks(range(9))
    ax.set_xlabel("Hamming distance between 64-bit perceptual hashes of two images")
    ax.set_ylabel("Image pairs (log scale)")
    ax.set_title("Near-duplicate candidates", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor="black", fontsize=8)
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------------------------------------ training
def _runs(cfg: TrainConfig):
    out = []
    for f in sorted(cfg.runs_dir.glob("*/run.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        out.append((f.parent, meta, pd.read_csv(f.parent / "metrics.csv")))
    return out


def fig_val_curves_and_stopping(cfg: TrainConfig, path: Path) -> Optional[Path]:
    """Validation Dice of every run with the selected epoch, and best vs last epoch of every run."""
    runs = _runs(cfg)
    if not runs:
        return None
    names, keys = _names(cfg), list(cfg.audit.datasets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white", gridspec_kw={"width_ratios": [1.5, 1]})
    for run_dir, meta, m in runs:
        c = COLORS[keys.index(meta["source"]) % 2]
        axes[0].plot(m["epoch"], m["val_dice"], color=c, alpha=0.55, linewidth=1.3)
        b = m.loc[m["val_dice"].idxmax()]
        axes[0].scatter([b["epoch"]], [b["val_dice"]], color=c, s=40, edgecolor="black", zorder=4)
        axes[1].scatter([meta["best_epoch"]], [meta["epochs_run"]], color=c, s=55, edgecolor="black", zorder=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Dice (mean per image)")
    axes[0].set_title("Validation Dice per run (dot = epoch selected for evaluation)", loc="left", fontweight="bold")
    mn = cfg.section("training").get("min_epochs", 0)
    lim = max(m["epoch"].max() for _, _, m in runs) + 2
    axes[1].plot([0, lim], [0, lim], color="#999999", linestyle=":")
    if mn:
        axes[1].axhline(mn, color="black", linestyle="--", linewidth=1)
        axes[1].text(2, mn + 1.5, f"min_epochs = {mn}", fontsize=8, color="black")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("Selected (best) epoch")
    axes[1].set_ylabel("Last epoch (where training stopped)")
    axes[1].set_title("Selected epoch vs stopping epoch", loc="left", fontweight="bold")
    for ax in axes:
        _style(ax)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[i % 2], markeredgecolor="black", markersize=8, label=f"{names[k]}-trained") for i, k in enumerate(keys)]
    axes[0].legend(handles=handles, frameon=False, labelcolor="black", fontsize=9, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def fig_train_val_gap(cfg: TrainConfig, path: Path) -> Optional[Path]:
    """Train Dice minus validation Dice over the epochs: how much of the training quality does not carry over to validation."""
    runs = _runs(cfg)
    if not runs:
        return None
    names, keys = _names(cfg), list(cfg.audit.datasets)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="white")
    for i, k in enumerate(keys):
        frames = [m.assign(gap=m["train_dice"] - m["val_dice"]) for _, meta, m in runs if meta["source"] == k]
        if not frames:
            continue
        allm = pd.concat(frames)
        g = allm.groupby("epoch")["gap"].agg(["mean", "min", "max"])
        ax.fill_between(g.index, g["min"], g["max"], color=COLORS[i % 2], alpha=0.18, linewidth=0)
        ax.plot(g.index, g["mean"], color=COLORS[i % 2], linewidth=2, label=f"{names[k]}-trained (mean over seeds, band = min-max)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Dice - validation Dice")
    ax.set_title("Gap between training and validation Dice", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor="black", fontsize=8)
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------------------------------------ results
def fig_dice_ecdf(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    """Empirical cumulative distribution of the per-image Dice (mean over seeds) for E1-E4."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor="white")
    for c in conditions(cfg):
        v = np.sort(df[df["eval_set"] == c.eval_set].groupby("stem")["dice"].mean().to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c.color, linewidth=2.2, linestyle="-" if c.own else "--", label=c.label.replace("\n", " "))
    ax.set_xlabel("Dice per image (mean over seeds)")
    ax.set_ylabel("Share of images with Dice at most x")
    ax.set_title("Cumulative distribution of the per-image Dice", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor="black", fontsize=7.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def fig_paired_same_images(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Optional[Path]:
    """The same test images segmented by the model trained on their dataset and by the model trained on the other dataset."""
    keys, names = list(cfg.audit.datasets), _names(cfg)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 5.4), facecolor="white", squeeze=False)
    for j, k in enumerate(keys):
        own = _seed_mean(df, f"{k}_test", ["dice"])
        ext = df[(df["eval_set"] == f"{k}_external") & (df["target_split"] == "test")].groupby("stem")["dice"].mean()
        joined = own.join(ext.rename("other"), how="inner")
        ax = axes[0][j]
        if joined.empty:
            ax.axis("off")
            continue
        ax.scatter(joined["dice"], joined["other"], s=20, alpha=0.6, color=COLORS[j % 2], linewidths=0)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.scatter([joined["dice"].mean()], [joined["other"].mean()], marker="D", s=70, color="white", edgecolor="black", zorder=5)
        better = float((joined["dice"] > joined["other"]).mean() * 100)
        ax.set_xlabel(f"Dice, model trained on {names[k]}")
        ax.set_ylabel(f"Dice, model trained on the other dataset")
        ax.set_title(f"{names[k]} test images (n = {len(joined)}): own model better on {better:.0f}%", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        _style(ax)
    fig.text(0.01, 0.005, "each dot: one test image, Dice averaged over seeds; diamond: mean; points below the diagonal favour the model trained on the image's own dataset", fontsize=8, color="black")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, path)


def fig_precision_recall(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    """Precision against recall of every seed for E1-E4, with lines of constant Dice (over- vs under-segmentation)."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="white")
    p = np.linspace(0.02, 1, 200)
    for d in (0.2, 0.4, 0.6, 0.8):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = d * p / (2 * p - d)
        ok = (r > 0) & (r <= 1) & (2 * p - d > 0)
        ax.plot(p[ok], r[ok], color="#bbbbbb", linewidth=1)
        ax.text(1.005, d / (2 - d), f"F={d:g}", fontsize=7, color="#555555", va="center")
    for c in conditions(cfg):
        g = df[df["eval_set"] == c.eval_set].groupby("seed")[["precision", "recall"]].mean()
        ax.scatter(g["precision"], g["recall"], s=55, marker="o" if c.own else "s", facecolor=c.color if c.own else "white", edgecolor=c.color, linewidth=1.8, label=c.label.replace("\n", " "))
    ax.set_xlim(0.1, 1.04)
    ax.set_ylim(0.1, 1.0)
    ax.set_xlabel("Precision (mean over the images of a set, one dot per seed)")
    ax.set_ylabel("Recall")
    ax.set_title("Precision and recall: over- vs under-segmentation", loc="left", fontweight="bold")
    ax.legend(frameon=False, labelcolor="black", fontsize=7.5, loc="lower left")
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------------------------------------ errors
def _binned(x: np.ndarray, y: np.ndarray, bins: int = 8):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    mids, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x <= b)
        if sel.sum() >= 3:
            mids.append(np.median(x[sel]))
            meds.append(np.median(y[sel]))
    return np.array(mids), np.array(meds)


def fig_dice_vs_size(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    """Per-image Dice against the ground-truth polyp area, with the median in equal-count bins."""
    conds = conditions(cfg)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), facecolor="white", sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), conds):
        g = _seed_mean(df, c.eval_set, ["dice", "gt_area"])
        x, y = g["gt_area"].to_numpy() * 100, g["dice"].to_numpy()
        ax.scatter(x, y, s=9, alpha=0.35, color=c.color, linewidths=0)
        mx, my = _binned(x, y)
        ax.plot(mx, my, color="black", linewidth=2, marker="o", markersize=4, label="median per bin")
        ax.set_xscale("log")
        ax.set_title(c.label.replace("\n", " "), loc="left", fontsize=9, fontweight="bold")
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Ground-truth polyp area (% of the image, log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dice per image")
    axes[0][0].legend(frameon=False, labelcolor="black", fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def fig_pred_vs_gt_area(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Path:
    """Predicted polyp area against the true area: does the model over- or under-segment, and does it miss small polyps?"""
    eps = 1e-4
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), facecolor="white", sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), conditions(cfg)):
        g = _seed_mean(df, c.eval_set, ["gt_area", "pred_area"])
        x, y = g["gt_area"].to_numpy(), g["pred_area"].to_numpy()
        ax.scatter(x, np.clip(y, eps, None), s=9, alpha=0.35, color=c.color, linewidths=0)
        ax.plot([eps, 1], [eps, 1], color="black", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(c.label.replace("\n", " "), loc="left", fontsize=9, fontweight="bold")
        ax.text(0.97, 0.04, f"no polyp predicted by any seed: {int((y == 0).sum())} images\npredicted area / true area: median {np.median(y[x > 0] / x[x > 0]):.2f}", transform=ax.transAxes, ha="right", fontsize=8, color="black")
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("True polyp area (fraction of the image, log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted area (seed mean; empty shown at 1e-4)")
    fig.tight_layout()
    return _save(fig, path)


def fig_dice_vs_brightness(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Optional[Path]:
    """Per-image Dice against the mean grey brightness of the image (a descriptive look at the brightness shift)."""
    tables = {k: _audit_table(cfg, k) for k in cfg.audit.datasets}
    if any(t is None for t in tables.values()):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), facecolor="white", sharey=True)
    for ax, c in zip(axes.ravel(), conditions(cfg)):
        target = c.eval_set.rsplit("_", 1)[0]
        g = _seed_mean(df, c.eval_set, ["dice"]).reset_index().merge(tables[target][["image_file", "brightness_gray"]], on="image_file", how="left")
        g = g.dropna(subset=["brightness_gray"])
        ax.scatter(g["brightness_gray"], g["dice"], s=9, alpha=0.35, color=c.color, linewidths=0)
        rho_text = ""
        if len(g) > 3:
            k, b = np.polyfit(g["brightness_gray"], g["dice"], 1)
            xs = np.linspace(g["brightness_gray"].min(), g["brightness_gray"].max(), 20)
            ax.plot(xs, k * xs + b, color="black", linewidth=1.8)
            rho, p = stats.spearmanr(g["brightness_gray"], g["dice"])
            rho_text = f"\nSpearman rho = {rho:.2f} (descriptive, n = {len(g)})"
        ax.set_title(c.label.replace("\n", " ") + rho_text, loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean grey level of the image (0-255)")
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Dice per image")
    fig.tight_layout()
    return _save(fig, path)


def fig_dice_by_sequence(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> Optional[Path]:
    """Dice per video sequence: how much the result depends on which sequences are evaluated."""
    seq_sets = [(c, df[df["eval_set"] == c.eval_set]) for c in conditions(cfg)]
    seq_sets = [(c, g) for c, g in seq_sets if g["sequence_id"].notna().any()]
    if not seq_sets:
        return None
    fig, axes = plt.subplots(len(seq_sets), 1, figsize=(13, 3.9 * len(seq_sets)), facecolor="white", squeeze=False)
    for ax, (c, g) in zip(axes.ravel(), seq_sets):
        s = g.groupby(["sequence_id", "stem"]).agg(dice=("dice", "mean"), part=("target_split", "first")).reset_index()
        order = sorted(s["sequence_id"].unique(), key=natural_key)
        for i, q in enumerate(order):
            sub = s[s["sequence_id"] == q]
            col = SPLIT_COLORS.get(sub["part"].iloc[0], c.color) if c.eval_set.endswith("_external") else c.color
            ax.scatter(np.full(len(sub), i) + np.linspace(-0.25, 0.25, len(sub)), sub["dice"], s=10, alpha=0.5, color=col, linewidths=0)
            ax.plot([i - 0.35, i + 0.35], [sub["dice"].mean()] * 2, color="black", linewidth=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=7)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Dice per image")
        ax.set_xlabel("Video sequence")
        ax.set_title(c.label.replace("\n", " ") + "  (black line: sequence mean)", loc="left", fontsize=10, fontweight="bold")
        _style(ax)
        if c.eval_set.endswith("_external"):
            ax.legend(handles=[plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SPLIT_COLORS[p], markersize=7, label=f"{p} part of its own dataset") for p in ("train", "val", "test")],
                      frameon=False, labelcolor="black", fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------------------------------------ robustness
def fig_original_vs_256(cfg: TrainConfig, path: Path) -> Optional[Path]:
    """Per-image Dice at the original resolution against the 256 x 256 Dice (robustness of the resizing)."""
    f = cfg.results_dir / "original_resolution" / "per_image_metrics_original.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
    axes[0].scatter(d["dice_256"], d["dice_original"], s=4, alpha=0.25, color="#555555", linewidths=0)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set_xlabel("Dice at 256 x 256")
    axes[0].set_ylabel("Dice at the original resolution")
    axes[0].set_title(f"{len(d):,} image-run values", loc="left", fontweight="bold")
    diff = d["dice_original"] - d["dice_256"]
    axes[1].hist(diff, bins=np.linspace(-0.05, 0.05, 61), color="#555555")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Dice (original) - Dice (256 x 256)")
    axes[1].set_ylabel("Image-run values (log scale)")
    axes[1].set_title(f"mean {diff.mean():+.4f}, largest absolute difference {diff.abs().max():.3f}", loc="left", fontweight="bold")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def fig_gap_by_seed_machine(cfg: TrainConfig, by_seed: pd.DataFrame, path: Path) -> Path:
    """Dice gap of each seed with the machine that trained it, to show that the machine does not drive the result."""
    d = by_seed[by_seed["metric"] == "dice"]
    dirs = directions(cfg)
    machines = {}
    for run in sorted(cfg.runs_dir.glob("*/run.json")):
        env = json.loads(run.read_text(encoding="utf-8"))["environment"]
        machines[(run.parent.name.split("_seed")[0], int(run.parent.name.split("_seed")[1]))] = env.get("gpu", env.get("device", "?")).replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
    kinds = sorted(set(machines.values()))
    markers = ["o", "^", "s", "D"]
    fig, ax = plt.subplots(figsize=(7.4, 5), facecolor="white")
    for i, (key, label, own, ext) in enumerate(dirs):
        g = d[d["direction"] == label]
        for _, r in g.iterrows():
            m = machines.get((key, int(r["seed"])), "?")
            ax.scatter([i + (int(r["seed"]) - 3) * 0.05], [r["gap"]], s=70, marker=markers[kinds.index(m) % 4] if m in kinds else "o", color=COLORS[i % 2], edgecolor="black", zorder=3)
            ax.text(i + (int(r["seed"]) - 3) * 0.05, r["gap"] + 0.006, str(int(r["seed"])), ha="center", fontsize=8, color="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels([x[1] for x in dirs])
    ax.set_xlim(-0.5, len(dirs) - 0.5)
    ax.set_ylabel("Dice gap = own test - external (number = seed)")
    ax.set_title("Gap per seed and training machine", loc="left", fontweight="bold")
    ax.legend(handles=[plt.Line2D([0], [0], marker=markers[i % 4], color="none", markerfacecolor="#cccccc", markeredgecolor="black", markersize=9, label=m) for i, m in enumerate(kinds)],
              frameon=False, labelcolor="black", fontsize=8)
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)
