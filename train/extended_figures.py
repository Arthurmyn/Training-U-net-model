"""Figures for the extended analysis (PNG, 200 dpi, English labels, black text, white background)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .config import TrainConfig  # noqa: E402
from .extended import load_extended, out_dir  # noqa: E402
from .extended_stats import directions, preliminary  # noqa: E402

COLORS = ["#2a78d6", "#eb6834"]            # validated categorical slots 1-2 (blue, orange); one colour per training dataset
DPI = 200


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def _shade(hex_color: str, k: int) -> tuple:
    """Three shades of one hue for seeds 1-3: darker, base, lighter (identity stays with the hue)."""
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(base * 0.7) if k == 0 else tuple(base) if k == 1 else tuple(base + (1 - base) * 0.45)


def _label(cfg: TrainConfig, key: str, text: str) -> str:
    tag = preliminary(cfg, key)
    return f"{text}\n{tag}" if tag else text


def figure_training_curves(cfg: TrainConfig, path: Path) -> None:
    """Train/val loss and Dice per epoch, one line per seed, one column per training dataset."""
    keys = list(cfg.audit.datasets)
    fig, axes = plt.subplots(2, len(keys), figsize=(5.6 * len(keys), 7.2), facecolor="white", squeeze=False)
    for c, key in enumerate(keys):
        color = COLORS[c % 2]
        runs = sorted(cfg.runs_dir.glob(f"{key}_seed*/metrics.csv"))
        for k, f in enumerate(runs):
            m = pd.read_csv(f)
            col = _shade(color, k)
            for r, (train_col, val_col) in enumerate((("train_loss", "val_loss"), ("train_dice", "val_dice"))):
                axes[r][c].plot(m["epoch"], m[train_col], color=col, linestyle="-", linewidth=1.6)
                axes[r][c].plot(m["epoch"], m[val_col], color=col, linestyle="--", linewidth=1.6)
        title = cfg.audit.datasets[key].name + "-trained"
        axes[0][c].set_title(_label(cfg, key, title), fontsize=11, fontweight="bold")
        axes[0][c].set_ylabel("Loss (BCE + soft Dice)")
        axes[1][c].set_ylabel("Dice (mean per image)")
        axes[1][c].set_xlabel("Epoch")
        axes[1][c].set_ylim(-0.02, 1.0)
        for r in range(2):
            _style(axes[r][c])
        handles = [Line2D([0], [0], color=_shade(color, k), lw=2, label=f"seed {k + 1}") for k in range(len(runs))] + \
                  [Line2D([0], [0], color="black", lw=2, linestyle="-", label="train"), Line2D([0], [0], color="black", lw=2, linestyle="--", label="validation")]
        axes[1][c].legend(handles=handles, frameon=False, loc="lower right", labelcolor="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def figure_gap(cfg: TrainConfig, by_seed: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Dice gap (own test minus external) per seed with the t-based 95% CI (df = 2), both directions."""
    d = by_seed[by_seed["metric"] == "dice"]
    s = summary[summary["metric"] == "dice"].reset_index(drop=True)
    deltas = [float(x) for x in cfg.section("analysis").get("practical_thresholds", [0.05])]
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor="white")
    for i, (key, label, own, ext) in enumerate(directions(cfg)):
        g = d[d["direction"] == label]
        r = s[s["direction"] == label].iloc[0]
        ax.scatter(np.full(len(g), i) + np.array([-0.08, 0, 0.08])[:len(g)], g["gap"], s=45, color=COLORS[i % 2], alpha=0.85, zorder=3, label="seed" if i == 0 else None)
        ax.errorbar([i + 0.25], [r["mean_gap"]], yerr=[[r["mean_gap"] - r["ci_t_low"]], [r["ci_t_high"] - r["mean_gap"]]], fmt="D", color="black",
                    markersize=7, capsize=6, linewidth=1.8, zorder=4, label="mean and t-based 95% CI (df = 2)" if i == 0 else None)
    ax.axhline(0, color="black", linewidth=1.0)
    for delta in deltas[:1]:
        ax.axhline(delta, color="#555555", linestyle=":", linewidth=1.2)
        ax.axhline(-delta, color="#555555", linestyle=":", linewidth=1.2)
        ax.text(1.62, delta, f"+{delta:g}", va="bottom", ha="right", fontsize=8, color="black")
        ax.text(1.62, -delta, f"-{delta:g}", va="top", ha="right", fontsize=8, color="black")
    ax.set_xticks(range(len(directions(cfg))))
    ax.set_xticklabels([_label(cfg, k, f"{l}\n({o} minus {e})") for k, l, o, e in directions(cfg)], fontsize=8)
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylabel("Dice gap = own test - external")
    ax.set_title("Dice gap per seed", loc="left", fontweight="bold")
    _style(ax)
    ax.legend(frameon=False, labelcolor="black", fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def figure_distributions(cfg: TrainConfig, df: pd.DataFrame, path: Path) -> None:
    """Per-image Dice, IoU, HD95 and Boundary IoU (mean over seeds per image) for the four conditions."""
    conds = []
    for i, (key, label, own, ext) in enumerate(directions(cfg)):
        conds += [(own, key, i, f"{label}\nown test"), (ext, key, i, f"{label}\nexternal")]
    metrics = [("dice", "Dice (higher is better)"), ("iou", "IoU (higher is better)"), ("hd95", "HD95, pixels (lower is better)"), ("boundary_iou", "Boundary IoU (higher is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), facecolor="white")
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        data, labels = [], []
        for pos, (eval_set, key, i, text) in enumerate(conds):
            v = df[df["eval_set"] == eval_set].groupby("stem")[metric].mean().to_numpy()
            data.append(v)
            labels.append(_label(cfg, key, text))
            bp = ax.boxplot(v, positions=[pos], widths=0.55, patch_artist=True, showfliers=False)
            col = COLORS[i % 2]
            rgb = matplotlib.colors.to_rgb(col)
            for b in bp["boxes"]:
                # own test: tinted fill; external: outline only (alpha must not be applied to the outline, so use an RGBA fill)
                b.set(facecolor=(*rgb, 0.45) if text.endswith("own test") else "none", edgecolor=col, linewidth=2)
            for part in ("whiskers", "caps", "medians"):
                for ln in bp[part]:
                    ln.set(color=col, linewidth=2)
            jit = np.random.default_rng(pos).uniform(-0.2, 0.2, len(v))
            ax.scatter(pos + jit, v, s=5, color=col, alpha=0.3, linewidths=0)
            ax.plot(pos, v.mean(), marker="D", color="white", markeredgecolor="black", markersize=6, zorder=5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        _style(ax)
    fig.text(0.01, 0.005, "box: quartiles and median, diamond: mean, dots: images (each averaged over 3 seeds); filled box = own test, open box = external", fontsize=8, color="black")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def run(cfg: TrainConfig) -> List[str]:
    """Write the three figures to results/extended/figures/."""
    fig_dir = out_dir(cfg) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_extended(cfg)
    figure_training_curves(cfg, fig_dir / "fig_a_training_curves.png")
    figure_gap(cfg, pd.read_csv(out_dir(cfg) / "gap_by_seed.csv"), pd.read_csv(out_dir(cfg) / "gap_summary.csv"), fig_dir / "fig_b_gap_per_seed_dice.png")
    figure_distributions(cfg, df, fig_dir / "fig_c_metric_distributions.png")
    return ["fig_a_training_curves.png", "fig_b_gap_per_seed_dice.png", "fig_c_metric_distributions.png"]
