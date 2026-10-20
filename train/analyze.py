"""Statistics over all evaluated runs: Dice per image, bootstrap confidence intervals, practical-threshold decisions.

``python -m train.analyze`` reads ``runs/*/per_image_dice.csv`` and writes ``results/``.

What this does and does not show: it measures how much Dice changes when a model is applied to the other
dataset (a combined effect of everything that differs between the datasets). It cannot say whether brightness,
resolution or annotation style is the cause.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import TrainConfig, load_train_config  # noqa: E402

log = logging.getLogger("train")
COLORS = ["#2a78d6", "#eb6834"]          # validated categorical slots 1-2 (blue, orange)


def load_results(runs_dir: Path) -> pd.DataFrame:
    """Concatenate per-image results of all evaluated runs."""
    files = sorted(runs_dir.glob("*/per_image_dice.csv"))
    if not files:
        raise FileNotFoundError(f"no per_image_dice.csv found under {runs_dir}; train and evaluate first")
    return pd.concat([pd.read_csv(f, dtype={"sequence_id": str}) for f in files], ignore_index=True)


def cluster_bootstrap_means(values: np.ndarray, clusters: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean when whole clusters (video sequences) are resampled.

    Frames of one sequence are almost identical, so they are not independent: resampling clusters, not
    frames, gives honest intervals. If every image is its own cluster this is the ordinary bootstrap.
    """
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def interval(draws: np.ndarray, level: float) -> Tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def decide(lo: float, hi: float, t: float) -> str:
    """Practical-significance reading of a CI for a drop in Dice."""
    if lo > t:
        return f"drop > {t:g} (CI entirely above)"
    if hi < t:
        return f"drop < {t:g} (CI entirely below)"
    return f"inconclusive for {t:g}"


def per_image_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average Dice over seeds for each (eval_set, image)."""
    g = df.groupby(["eval_set", "source", "kind", "target", "stem"], as_index=False).agg(
        dice=("dice", "mean"), n_seeds=("seed", "nunique"), sequence_id=("sequence_id", "first"), target_split=("target_split", "first"))
    g["cluster"] = g["sequence_id"].fillna(g["stem"]).astype(str)
    return g


def analyze(cfg: TrainConfig) -> Path:
    """Run the analysis and write results/summary.md, results/tables/*.csv and the figure."""
    a = cfg.section("analysis")
    n_boot, level = int(a.get("bootstrap_samples", 10000)), float(a.get("ci_level", 0.95))
    thresholds: List[float] = [float(t) for t in a.get("practical_thresholds", [0.05, 0.03, 0.10])]
    rng = np.random.default_rng(int(a.get("seed", 42)))
    df = load_results(cfg.runs_dir)
    pim = per_image_means(df)
    names = {k: v.name for k, v in cfg.audit.datasets.items()}
    out = cfg.results_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # ---- Table 1: Dice per condition
    conds: Dict[str, dict] = {}
    for es, g in pim.groupby("eval_set"):
        draws = cluster_bootstrap_means(g["dice"].to_numpy(), g["cluster"].to_numpy(), n_boot, rng)
        seed_means = df[df["eval_set"] == es].groupby("seed")["dice"].mean()
        lo, hi = interval(draws, level)
        src, kind = g["source"].iloc[0], g["kind"].iloc[0]
        conds[es] = {"draws": draws, "mean": float(g["dice"].mean()), "lo": lo, "hi": hi}
        conds[es].update(source=src, kind=kind)
        conds[es]["row"] = {"trained on": names[src], "evaluated on": names[g["target"].iloc[0]], "condition": "in-domain test" if kind == "in_domain" else "external (whole other dataset)",
                            "eval_set": es, "images": len(g), "clusters": g["cluster"].nunique(), "seeds": int(g["n_seeds"].max()),
                            "mean Dice": g["dice"].mean(), f"CI{int(level * 100)} low": lo, f"CI{int(level * 100)} high": hi,
                            "SD across seeds": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else float("nan")}
    t1 = pd.DataFrame([c["row"] for c in conds.values()]).sort_values(["trained on", "condition"]).reset_index(drop=True)
    t1.to_csv(out / "tables" / "dice_by_condition.csv", index=False, float_format="%.4f")

    # ---- Table 2: drop in Dice when the model meets the other dataset (unpaired: different images)
    rows2 = []
    for src in cfg.sources:
        ins = next((e for e, c in conds.items() if c["source"] == src and c["kind"] == "in_domain" and e.endswith("_test")), None)
        ext = next((e for e, c in conds.items() if c["source"] == src and c["kind"] == "external"), None)
        if ins is None or ext is None:
            continue
        d = conds[ins]["draws"] - conds[ext]["draws"]
        lo, hi = interval(d, level)
        row = {"trained on": names[src], "in-domain Dice": conds[ins]["mean"], "external Dice": conds[ext]["mean"],
               "drop (in-domain - external)": conds[ins]["mean"] - conds[ext]["mean"], f"CI{int(level * 100)} low": lo, f"CI{int(level * 100)} high": hi}
        for t in thresholds:
            row[f"reading at {t:g}"] = decide(lo, hi, t)
        rows2.append(row)
    t2 = pd.DataFrame(rows2)
    t2.to_csv(out / "tables" / "dice_drop_external.csv", index=False, float_format="%.4f")

    # ---- Table 3: same target images, different training dataset (paired)
    rows3 = []
    for tgt in cfg.sources:
        own = pim[(pim["eval_set"] == f"{tgt}_test")]
        foreign = pim[(pim["target"] == tgt) & (pim["kind"] == "external") & (pim["target_split"] == "test")]
        if own.empty or foreign.empty:
            continue
        m = own.merge(foreign, on="stem", suffixes=("_own", "_foreign"))
        if m.empty:
            continue
        diff = (m["dice_own"] - m["dice_foreign"]).to_numpy()
        draws = cluster_bootstrap_means(diff, m["cluster_own"].to_numpy(), n_boot, rng)
        lo, hi = interval(draws, level)
        row = {"target (test images)": names[tgt], "images": len(m), "clusters": m["cluster_own"].nunique(),
               "Dice, trained on target": m["dice_own"].mean(), "Dice, trained on other": m["dice_foreign"].mean(),
               "difference (own - other)": float(diff.mean()), f"CI{int(level * 100)} low": lo, f"CI{int(level * 100)} high": hi}
        for t in thresholds:
            row[f"reading at {t:g}"] = decide(lo, hi, t)
        rows3.append(row)
    t3 = pd.DataFrame(rows3)
    t3.to_csv(out / "tables" / "paired_same_target_images.csv", index=False, float_format="%.4f")

    # ---- per-image means for reuse
    pim.drop(columns=["cluster"]).to_csv(out / "tables" / "per_image_mean_dice.csv", index=False, float_format="%.6f")

    _figure(pim, names, out / "fig_dice_distributions.png")
    (out / "summary.md").write_text(_markdown(cfg, t1, t2, t3, n_boot, level, thresholds), encoding="utf-8")
    log.info("wrote %s", out)
    return out / "summary.md"


def _figure(pim: pd.DataFrame, names: Dict[str, str], path: Path) -> None:
    order = sorted(pim["eval_set"].unique(), key=lambda e: (pim[pim["eval_set"] == e]["source"].iloc[0], pim[pim["eval_set"] == e]["kind"].iloc[0] != "in_domain"))
    sources = sorted(pim["source"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, es in enumerate(order):
        g = pim[pim["eval_set"] == es]
        col = COLORS[sources.index(g["source"].iloc[0]) % 2]
        inside = g["kind"].iloc[0] == "in_domain"
        bp = ax.boxplot(g["dice"], positions=[i], widths=0.55, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=col if inside else "none", edgecolor=col, linewidth=2, alpha=0.55 if inside else 1)
        for k in ("whiskers", "caps", "medians"):
            for ln in bp[k]:
                ln.set(color=col, linewidth=2)
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(g))
        ax.scatter(i + jitter, g["dice"], s=8, color=col, alpha=0.35, linewidths=0)
        ax.plot(i, g["dice"].mean(), marker="D", color="white", markeredgecolor=col, markersize=7, zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"trained on {names[pim[pim['eval_set'] == e]['source'].iloc[0]]}\ntested on {names[pim[pim['eval_set'] == e]['target'].iloc[0]]}\n({'in-domain' if pim[pim['eval_set'] == e]['kind'].iloc[0] == 'in_domain' else 'external'})" for e in order], fontsize=8)
    ax.set_ylabel("Dice per image (mean over seeds)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Dice per image: in-domain vs external", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e4e3df")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "box: quartiles, line: median, diamond: mean, dots: images", fontsize=7, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "_not available (needs the runs of both datasets)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _markdown(cfg, t1, t2, t3, n_boot, level, thresholds) -> str:
    return "\n".join([
        "# Results", "",
        f"Dice per image at probability threshold {cfg.section('training').get('threshold', 0.5)}, computed on {cfg.image_size}x{cfg.image_size} preprocessed masks. "
        f"Confidence intervals: {n_boot} bootstrap samples, {int(level * 100)}%, whole video sequences resampled where sequence IDs exist (frames of one sequence are not independent), otherwise images. "
        f"Seeds are averaged per image first; the spread between seeds is the last column of table 1.", "",
        "## 1. Dice by condition", "", _fmt(t1), "",
        "## 2. Drop in Dice when a model is applied to the other dataset", "",
        "Drop = in-domain test Dice minus Dice on the whole other dataset (different images, so the two sets are resampled independently).", "", _fmt(t2), "",
        "## 3. Same target images, different training dataset (paired)", "",
        "Dice of the model trained on the target dataset versus the model trained on the other dataset, on the same test images of the target dataset.", "", _fmt(t3), "",
        f"Reading of an interval against a practical threshold t (main {thresholds[0]:g}, sensitivity {', '.join(f'{t:g}' for t in thresholds[1:])}): "
        "'drop > t' if the whole interval lies above t, 'drop < t' if it lies below, otherwise inconclusive. These are decision rules chosen for this pipeline; check them against the hypotheses in the methodology.", "",
        "## What these numbers do not show", "",
        "- **Statistical uncertainty** is what the intervals describe: how much the result would vary with other images (and sequences).",
        "- **Practical significance** is the comparison with the threshold t, decided before looking at results.",
        "- **Cause** is not established. The experiment measures the combined effect of moving between datasets (brightness, resolution, endoscope, annotation style, patient population, number of training images). It cannot attribute a Dice drop to any single factor.",
        "- CVC-ClinicDB splits are by video sequence, not by patient, so some patient-level overlap between train and test cannot be excluded (see outputs/audit_report.md).", "",
        "Figure: `fig_dice_distributions.png`. Tables: `tables/`.", ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate all evaluated runs into tables and a figure.")
    ap.add_argument("--config", default="train_config.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    analyze(load_train_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
