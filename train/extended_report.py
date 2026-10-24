"""results/extended/METRICS.md (definitions) and README.md (file list, acceptance checks, numbers for the report)."""
from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from .config import TrainConfig
from .extended import error_dir, load_extended, out_dir
from .extended_stats import METRICS, preliminary

METRICS_MD = """# Metric definitions

All masks are binary (polyp / background) at the model resolution, 256 x 256 pixels, produced exactly as in the original
evaluation: the audited preprocessing, the best checkpoint of each run (by validation Dice), eval mode, no augmentation,
probability threshold 0.5. Distances are in pixels. TP, FP, FN are pixel counts of the prediction against the ground truth.

| Metric | Definition |
|---|---|
| Dice | 2 TP / (2 TP + FP + FN). If prediction and ground truth are both empty, Dice = 1 (as in the existing code). |
| IoU | TP / (TP + FP + FN). Both empty: 1. Always equals Dice / (2 - Dice). |
| precision | TP / (TP + FP); 0 if no pixel is predicted. |
| recall | TP / (TP + FN); 0 if the ground truth is empty (does not occur in these datasets). |
| boundary of a mask M | M AND NOT erosion(M); 4-connectivity (cross-shaped structuring element), `border_value = 0`, so pixels on the image border count as boundary. |
| HD95 | max( P95 of the distances from boundary(prediction) to boundary(ground truth), P95 of the distances from boundary(ground truth) to boundary(prediction) ). Distances come from `scipy.ndimage.distance_transform_edt`; percentiles use `np.percentile` with default (linear) interpolation. **Empty prediction: HD95 = 256 * sqrt(2) = 362.04** (the image diagonal) and `miss = 1`. Both masks empty: 0. Empty ground truth with a non-empty prediction is undefined and gets 362.04 too (does not occur here). |
| Boundary IoU | Band B_d(M) = pixels of M whose distance to the nearest background pixel is <= d, computed on M padded with a 1-pixel zero border, d = 5 pixels (`extended.boundary_iou_d` in `train_config.yaml`). BIoU = \\|B_d(pred) AND B_d(gt)\\| / \\|B_d(pred) OR B_d(gt)\\|. Empty prediction: 0. For masks narrower than 2d the band equals the mask, so BIoU equals IoU. |
| miss | 1 if the prediction contains no polyp pixel, else 0 (its mean over images is the miss rate). |

## Statistics

- **Seed-level mean** M_s: mean of a metric over the images of one evaluation set, for seed s. "Mean over seeds" and the sample SD (ddof = 1) are taken over the {n} values M_s.
- **Gap per seed** G_s = M_s(own test) - M_s(external): Kvasir-trained: kvasir_test minus cvc_external; CVC-trained: cvc_test minus kvasir_external. For HD95 and miss lower is better, so a negative gap means the model does better on its own data; `better` states the direction in every table.
- **Primary CI:** mean(G_s) +/- t * SD(G_s) / sqrt({n}), with t = t.ppf(0.975, {df}) = {t:.3f} (df = {df}).
- **Secondary CI:** cluster bootstrap (10 000 resamples, 95%, seed 42, `train/analyze.py`): images are averaged over seeds, whole video sequences are resampled for CVC frames, single images for Kvasir; own and external sets are resampled independently.
- **Decision rule (Dice only)** on a CI [L, U] and a practical threshold delta in {0.05 (main), 0.03, 0.10}: L > delta: *meaningful drop*; U < -delta: *meaningful increase*; [L, U] inside [-delta, delta]: *equivalence*; otherwise *inconclusive*.
- **Spearman correlations** (descriptive): seed-mean Dice per image on the external set against (a) the ground-truth area fraction at 256 px and (b) the mean grey brightness from `outputs/per_image_stats_*.csv`. p-values are descriptive: frames of one video sequence are not independent.
- **median_over_images** (in `metrics_by_condition.csv`) is the mean over seeds of the per-seed median over images; it is added because the mean HD95 is dominated by missed polyps (362.04 each).

## Preliminary results

CVC-trained models are under-trained (error E10 in `docs/errors_log.md`). Every table row and figure for them carries the label **preliminary (E10)**.
"""


def md_table(df: pd.DataFrame, floats: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def pm(mean: float, sd: float, metric: str) -> str:
    d = 2 if metric == "hd95" else 3
    return f"{mean:.{d}f} +/- {sd:.{d}f}"


def run(cfg: TrainConfig) -> List[str]:
    """Write METRICS.md and README.md from the CSV/JSON files of the previous stages."""
    out = out_dir(cfg)
    mbc = pd.read_csv(out / "metrics_by_condition.csv")
    n_seeds = int(mbc["n_seeds"].max())
    t_crit = float(stats.t.ppf(0.975, n_seeds - 1))
    metrics_md = METRICS_MD.replace("{n}", str(n_seeds)).replace("{df}", str(n_seeds - 1)).replace("{t:.3f}", f"{t_crit:.3f}")
    (out / "METRICS.md").write_text(metrics_md, encoding="utf-8")
    gs = pd.read_csv(out / "gap_summary.csv")
    dec = pd.read_csv(out / "decision_rule.csv")
    sp = pd.read_csv(out / "spearman.csv")
    acc = json.loads((out / "acceptance_checks.json").read_text(encoding="utf-8"))
    sel_path = error_dir(cfg) / "selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.is_file() else {}
    label = cfg.section("extended").get("preliminary_label", "preliminary (E10)")

    cond_rows = []
    for (eval_set, status), g in mbc.groupby(["eval_set", "status"], sort=False, dropna=False):
        status = "" if pd.isna(status) else status
        row = {"evaluation set": eval_set + (f" ({status})" if status else ""), "images": int(g["n_images"].iloc[0])}
        for m in METRICS:
            r = g[g["metric"] == m].iloc[0]
            row[m] = pm(r["mean_over_seeds"], r["sd_over_seeds"], m)
        cond_rows.append(row)
    cond = pd.DataFrame(cond_rows)

    gap = gs.copy()
    gap["direction"] = gap.apply(lambda r: r["direction"] + (f" ({r['status']})" if isinstance(r["status"], str) and r["status"] else ""), axis=1)
    gap_t = gap[["direction", "metric", "better", "mean_gap", "sd_gap", "ci_t_low", "ci_t_high", "ci_boot_low", "ci_boot_high"]]
    dec_t = dec[["direction", "delta", "mean_gap", "ci_t_low", "ci_t_high", "decision_t_ci", "ci_boot_low", "ci_boot_high", "decision_bootstrap_ci"]].copy()
    dec_t["direction"] = dec_t["direction"] + np.where(dec["status"].notna(), " (" + dec["status"].fillna("") + ")", "")
    a, b = acc["A_dice_reproduces_original"], acc["B_iou_equals_dice_over_2_minus_dice"]
    a_rows = pd.DataFrame([{"evaluation set": k, "images": v["n"], "mean abs diff": v["mean_abs_diff"], "max abs diff": v["max_abs_diff"], "share of images with diff < 0.01": v["share_below_0.01"]}
                           for k, v in a["by_eval_set"].items()])
    n_runs = len(list(cfg.runs_dir.glob("*/run.json")))
    n_eval = int(load_extended(cfg)[["run", "eval_set"]].drop_duplicates().shape[0])
    n_datasets = len(cfg.audit.datasets)
    protocol_seeds = int(cfg.section("extended").get("protocol_seeds", 5))
    planned_runs = n_datasets * protocol_seeds

    text = f"""# Extended analysis

Extended per-image metrics, seed statistics, error-analysis material and figures, computed from the saved weights
(`runs/*/best.pt`) without retraining and without changing any file under `runs/`, `outputs/` or the original `results/` files.

**Status:** CVC-trained models are under-trained (error E10, see `docs/errors_log.md`); all their results are **{label}**.

## Files

| File | Content |
|---|---|
| `per_image_metrics_extended.csv` | existing columns plus tp, fp, fn, iou, precision, recall, hd95, boundary_iou, miss (one row per run, evaluation set and image) |
| `METRICS.md` | definitions of all metrics and statistics |
| `acceptance_checks.json` | checks A and B (below) |
| `metrics_by_condition.csv` | seed-level means, mean over seeds, sample SD, median over images, per evaluation set and metric |
| `gap_by_seed.csv`, `gap_summary.csv` | gap per seed; mean gap, SD, t-based 95% CI (df = seeds - 1) and cluster-bootstrap 95% CI |
| `decision_rule.csv` | decision rule for Dice, delta 0.05 (main), 0.03, 0.10, on the t-based and the bootstrap CI |
| `spearman.csv` | descriptive Spearman correlations of external Dice with polyp size and brightness |
| `figures/` | training curves, Dice gap per seed, metric distributions |
| `../../error_analysis/` | frames, five-panel figures, blank annotation sheets, category rules |

## Reproduce
```bash
python -m train.extended inference    # about 10-15 minutes on CPU; stops if acceptance check A or B fails
python -m train.extended stats
python -m train.extended figures
python -m train.extended errors
python -m train.extended report
python -m pytest -q
```

## Acceptance checks

**Check A** (recomputed Dice reproduces `runs/*/per_image_dice.csv`; required: per-evaluation-set mean absolute difference < 0.002 and at least 99% of images with a difference < 0.01): **{'PASSED' if a['passed'] else 'FAILED'}** (unmatched rows: {a['unmatched_rows']}).

{md_table(a_rows, 5)}

**Check B** (IoU equals Dice / (2 - Dice) within 1e-4): **{'PASSED' if b['passed'] else 'FAILED'}**, maximum deviation {b['max_abs_deviation']:.2e}.

## Numbers for the report

Runs completed: **{n_runs} of {planned_runs} planned** by the protocol ({n_datasets} training datasets x {protocol_seeds} seeds). Evaluations completed: **{n_eval} in total ({n_runs} validation + {n_runs} own-test + {n_runs} external); {n_runs * 2} of {planned_runs * 2} planned counting own-test and external evaluations.**
This folder contains {n_seeds} seeds per training dataset.

### Mean +/- SD over the {n_seeds} seeds
{md_table(cond)}

HD95 in pixels (lower is better); miss is the share of images with an empty prediction (lower is better); all other metrics: higher is better.

### Gap = own test minus external (per seed, then summarised)
Primary CI: t-based, df = {n_seeds - 1} (t = {t_crit:.3f}). Secondary CI: cluster bootstrap.

{md_table(gap_t)}

### Decision rule (Dice)
{md_table(dec_t)}

### Spearman correlations (descriptive)
{md_table(sp[['trained_on', 'evaluated_on', 'variable', 'n_images', 'spearman_rho', 'p_value', 'status']])}

### Error analysis
{json.dumps(sel, indent=2) if sel else 'not generated yet (python -m train.extended errors)'}

## Caveats
- {n_seeds} seeds per dataset: the t multiplier is {t_crit:.3f}{', so intervals are wide' if n_seeds < 5 else ''}.
- The CVC test set has 91 frames from 5 video sequences; the bootstrap CI for CVC resamples these 5 sequences.
- CVC-ClinicDB is split by video sequence, not by patient (see `outputs/audit_report.md`).
- An effect measured here is the combined effect of everything that differs between the datasets; it is not attributed to a single factor.
"""
    (out / "README.md").write_text(text, encoding="utf-8")
    return ["METRICS.md", "README.md"]
