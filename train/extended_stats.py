"""Statistics for every metric over the 3 seeds: metrics by condition, gap per seed, t-based and bootstrap CIs,
decision rule (Dice only) and descriptive Spearman correlations."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .analyze import cluster_bootstrap_means, interval
from .config import TrainConfig
from .extended import EXTENDED_FILE, load_extended, out_dir

METRICS = ["dice", "iou", "precision", "recall", "hd95", "boundary_iou", "miss"]
HIGHER_IS_BETTER = {"dice": True, "iou": True, "precision": True, "recall": True, "boundary_iou": True, "hd95": False, "miss": False}
DEFAULT_DELTAS = [0.05, 0.03, 0.10]            # main threshold first


def preliminary(cfg: TrainConfig, source: str) -> str:
    """Label attached to results of under-trained models (CVC-trained, error E10)."""
    e = cfg.section("extended")
    return e.get("preliminary_label", "preliminary (E10)") if source in e.get("preliminary_sources", ["cvc"]) else ""


def directions(cfg: TrainConfig) -> List[Tuple[str, str, str, str]]:
    """(source key, label, own test set, external set) for every training dataset."""
    out = []
    for src, ds in cfg.audit.datasets.items():
        others = [k for k in cfg.audit.datasets if k != src]
        out.append((src, f"{ds.name}-trained", f"{src}_test", f"{others[0]}_external"))
    return out


def t_interval(values: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    """Mean, sample SD (ddof=1) and t-based CI with df = n - 1 (n = number of seeds; t = 4.303 for 3 seeds)."""
    v = np.asarray(values, float)
    n = len(v)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else float("nan")
    t = float(stats.t.ppf(1 - (1 - level) / 2, n - 1)) if n > 1 else float("nan")
    half = t * sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "t_crit": t, "low": mean - half, "high": mean + half}


def decide_delta(low: float, high: float, delta: float) -> str:
    """Decision rule on a CI [low, high] for a gap (own test minus external), practical threshold delta."""
    if low > delta:
        return "meaningful drop"
    if high < -delta:
        return "meaningful increase"
    if low >= -delta and high <= delta:
        return "equivalence"
    return "inconclusive"


def seed_means(df: pd.DataFrame, eval_set: str, metric: str) -> pd.Series:
    """Mean of ``metric`` over the images of ``eval_set`` for each seed."""
    return df[df["eval_set"] == eval_set].groupby("seed")[metric].mean().sort_index()


def metrics_by_condition(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """One row per (evaluation set, metric): seed-level means, mean over seeds, sample SD, median over images."""
    names = {k: v.name for k, v in cfg.audit.datasets.items()}
    rows = []
    for (source, eval_set), g in df.groupby(["source", "eval_set"], sort=True):
        for metric in METRICS:
            sm = seed_means(df, eval_set, metric)
            med = g.groupby("seed")[metric].median().mean()
            row = {"train_dataset": names[source], "test_dataset": names[g["target"].iloc[0]], "eval_set": eval_set,
                   "condition": "own test/val" if g["kind"].iloc[0] == "in_domain" else "external",
                   "status": preliminary(cfg, source), "metric": metric,
                   "better": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                   "n_images": int(g["stem"].nunique()), "n_seeds": int(sm.size)}
            row.update({f"seed_{s}": float(v) for s, v in sm.items()})
            row.update({"mean_over_seeds": float(sm.mean()), "sd_over_seeds": float(sm.std(ddof=1)),
                        "median_over_images_mean_over_seeds": float(med)})
            rows.append(row)
    return pd.DataFrame(rows)


def per_image_seed_mean(df: pd.DataFrame, eval_set: str, metric: str) -> pd.DataFrame:
    """Metric averaged over seeds for each image of the set, with the cluster used for the bootstrap (sequence or image)."""
    g = df[df["eval_set"] == eval_set].groupby("stem", sort=True).agg(value=(metric, "mean"), sequence_id=("sequence_id", "first")).reset_index()
    g["cluster"] = g["sequence_id"].where(g["sequence_id"].notna(), g["stem"]).astype(str)
    return g


def gap_tables(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gap per seed G_s = M_s(own test) - M_s(external) and its summary (t-based CI primary, cluster bootstrap secondary)."""
    a = cfg.section("analysis")
    n_boot, level, seed0 = int(a.get("bootstrap_samples", 10000)), float(a.get("ci_level", 0.95)), int(a.get("seed", 42))
    by_seed, summary = [], []
    for d_idx, (src, label, own, ext) in enumerate(directions(cfg)):
        if own not in set(df["eval_set"]) or ext not in set(df["eval_set"]):
            continue
        for metric in METRICS:
            mo, me = seed_means(df, own, metric), seed_means(df, ext, metric)
            gaps = (mo - me).dropna()
            for s in gaps.index:
                by_seed.append({"direction": label, "metric": metric, "better": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                                "seed": int(s), "own_test": own, "external": ext, "own_mean": float(mo[s]), "external_mean": float(me[s]),
                                "gap": float(gaps[s]), "status": preliminary(cfg, src)})
            t = t_interval(gaps.to_numpy(), level)
            po, pe = per_image_seed_mean(df, own, metric), per_image_seed_mean(df, ext, metric)
            # identical resample indices for every metric: the generator is re-created with the same entropy
            draws_own = cluster_bootstrap_means(po["value"].to_numpy(), po["cluster"].to_numpy(), n_boot, np.random.default_rng([seed0, d_idx, 0]))
            draws_ext = cluster_bootstrap_means(pe["value"].to_numpy(), pe["cluster"].to_numpy(), n_boot, np.random.default_rng([seed0, d_idx, 1]))
            blo, bhi = interval(draws_own - draws_ext, level)
            summary.append({"direction": label, "metric": metric, "better": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                            "own_test": own, "external": ext, "n_seeds": t["n"], "own_mean": float(mo.mean()), "external_mean": float(me.mean()),
                            "mean_gap": t["mean"], "sd_gap": t["sd"], "t_crit": t["t_crit"], "ci_t_low": t["low"], "ci_t_high": t["high"],
                            "ci_boot_low": blo, "ci_boot_high": bhi, "boot_samples": n_boot, "boot_clusters_own": int(po["cluster"].nunique()),
                            "boot_clusters_external": int(pe["cluster"].nunique()), "status": preliminary(cfg, src)})
    return pd.DataFrame(by_seed), pd.DataFrame(summary)


def decision_table(gap_summary: pd.DataFrame, deltas: List[float]) -> pd.DataFrame:
    """Decision rule for Dice only, on the t-based CI (primary) and the bootstrap CI (secondary)."""
    rows = []
    for r in gap_summary[gap_summary["metric"] == "dice"].itertuples():
        for delta in deltas:
            rows.append({"direction": r.direction, "delta": delta, "main_delta": delta == deltas[0], "mean_gap": r.mean_gap,
                         "ci_t_low": r.ci_t_low, "ci_t_high": r.ci_t_high, "decision_t_ci": decide_delta(r.ci_t_low, r.ci_t_high, delta),
                         "ci_boot_low": r.ci_boot_low, "ci_boot_high": r.ci_boot_high,
                         "decision_bootstrap_ci": decide_delta(r.ci_boot_low, r.ci_boot_high, delta), "status": r.status})
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Spearman rho (+ p, descriptive) between seed-mean Dice on the external set and (a) GT area fraction, (b) mean brightness."""
    rows = []
    for src, label, own, ext in directions(cfg):
        g = df[df["eval_set"] == ext]
        if g.empty:
            continue
        per = g.groupby("image_file").agg(dice=("dice", "mean"), gt_area_fraction=("gt_area", "first")).reset_index()
        target = ext.rsplit("_", 1)[0]
        stats_path = cfg.audit.output_dir / f"per_image_stats_{target}.csv"
        bright = pd.read_csv(stats_path)[["image_file", "brightness_gray"]]
        per = per.merge(bright, on="image_file", how="left", validate="one_to_one")
        for var in ("gt_area_fraction", "brightness_gray"):
            rho, p = stats.spearmanr(per["dice"], per[var])
            rows.append({"trained_on": label, "evaluated_on": cfg.audit.datasets[target].name, "eval_set": ext, "variable": var,
                         "n_images": int(per[var].notna().sum()), "spearman_rho": float(rho), "p_value": float(p), "status": preliminary(cfg, src),
                         "note": "descriptive only; frames of one video sequence are not independent"})
    return pd.DataFrame(rows)


def run(cfg: TrainConfig) -> List[str]:
    """Write metrics_by_condition.csv, gap_by_seed.csv, gap_summary.csv, decision_rule.csv and spearman.csv."""
    df = load_extended(cfg)
    out = out_dir(cfg)
    deltas = [float(x) for x in cfg.section("analysis").get("practical_thresholds", DEFAULT_DELTAS)]
    mbc = metrics_by_condition(df, cfg)
    by_seed, summ = gap_tables(df, cfg)
    tables = {"metrics_by_condition.csv": mbc, "gap_by_seed.csv": by_seed, "gap_summary.csv": summ,
              "decision_rule.csv": decision_table(summ, deltas), "spearman.csv": spearman_table(df, cfg)}
    for name, t in tables.items():
        t.to_csv(out / name, index=False, float_format="%.6f")
    return list(tables)
