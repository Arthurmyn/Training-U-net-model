# Extended analysis

Extended per-image metrics, seed statistics, error-analysis material and figures, computed from the saved weights
(`runs/*/best.pt`) without retraining and without changing any file under `runs/`, `outputs/` or the original `results/` files.

**Status:** CVC-trained models are under-trained (error E10, see `docs/errors_log.md`); all their results are **preliminary (E10)**.

## Files

| File | Content |
|---|---|
| `per_image_metrics_extended.csv` | existing columns plus tp, fp, fn, iou, precision, recall, hd95, boundary_iou, miss (one row per run, evaluation set and image) |
| `METRICS.md` | definitions of all metrics and statistics |
| `acceptance_checks.json` | checks A and B (below) |
| `metrics_by_condition.csv` | seed-level means, mean over seeds, sample SD, median over images, per evaluation set and metric |
| `gap_by_seed.csv`, `gap_summary.csv` | gap per seed; mean gap, SD, t-based 95% CI (df = 2) and cluster-bootstrap 95% CI |
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

**Check A** (recomputed Dice reproduces `runs/*/per_image_dice.csv`; required: per-evaluation-set mean absolute difference < 0.002 and at least 99% of images with a difference < 0.01): **PASSED** (unmatched rows: 0).

| evaluation set | images | mean abs diff | max abs diff | share of images with diff < 0.01 |
|---|---|---|---|---|
| cvc_external | 1836 | 0.00007 | 0.00191 | 1.00000 |
| cvc_test | 273 | 0.00003 | 0.00070 | 1.00000 |
| cvc_val | 276 | 0.00007 | 0.00221 | 1.00000 |
| kvasir_external | 3000 | 0.00002 | 0.00039 | 1.00000 |
| kvasir_test | 300 | 0.00003 | 0.00042 | 1.00000 |
| kvasir_val | 300 | 0.00004 | 0.00120 | 1.00000 |

**Check B** (IoU equals Dice / (2 - Dice) within 1e-4): **PASSED**, maximum deviation 2.22e-16.

## Numbers for the report

Runs completed: **6 of 10 planned** (as stated in the assignment). Evaluations completed: **18 in total (6 validation + 6 own-test + 6 external); 12 of 20 planned counting own-test and external evaluations.**
In this repository the plan was 2 training datasets x 3 seeds = 6 runs (`train_config.yaml`).

### Mean +/- SD over the 3 seeds
| evaluation set | images | dice | iou | precision | recall | hd95 | boundary_iou | miss |
|---|---|---|---|---|---|---|---|---|
| cvc_test (preliminary (E10)) | 91 | 0.378 +/- 0.036 | 0.288 +/- 0.031 | 0.463 +/- 0.065 | 0.411 +/- 0.021 | 136.15 +/- 10.12 | 0.078 +/- 0.018 | 0.212 +/- 0.023 |
| cvc_val (preliminary (E10)) | 92 | 0.467 +/- 0.046 | 0.351 +/- 0.043 | 0.454 +/- 0.073 | 0.649 +/- 0.005 | 90.62 +/- 13.85 | 0.083 +/- 0.029 | 0.080 +/- 0.066 |
| kvasir_external (preliminary (E10)) | 1000 | 0.397 +/- 0.007 | 0.277 +/- 0.004 | 0.326 +/- 0.011 | 0.780 +/- 0.054 | 113.18 +/- 3.74 | 0.062 +/- 0.007 | 0.001 +/- 0.002 |
| cvc_external | 612 | 0.655 +/- 0.036 | 0.562 +/- 0.031 | 0.735 +/- 0.016 | 0.698 +/- 0.054 | 74.52 +/- 6.24 | 0.272 +/- 0.021 | 0.063 +/- 0.036 |
| kvasir_test | 100 | 0.822 +/- 0.003 | 0.739 +/- 0.004 | 0.849 +/- 0.010 | 0.845 +/- 0.013 | 41.61 +/- 3.37 | 0.374 +/- 0.015 | 0.007 +/- 0.006 |
| kvasir_val | 100 | 0.854 +/- 0.010 | 0.778 +/- 0.011 | 0.909 +/- 0.017 | 0.846 +/- 0.006 | 39.59 +/- 5.30 | 0.417 +/- 0.014 | 0.013 +/- 0.012 |

HD95 in pixels (lower is better); miss is the share of images with an empty prediction (lower is better); all other metrics: higher is better.

### Gap = own test minus external (per seed, then summarised)
Primary CI: t-based, df = 2 (t = 4.303). Secondary CI: cluster bootstrap.

| direction | metric | better | mean_gap | sd_gap | ci_t_low | ci_t_high | ci_boot_low | ci_boot_high |
|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | dice | higher | 0.167 | 0.035 | 0.079 | 0.255 | 0.087 | 0.251 |
| Kvasir-SEG-trained | iou | higher | 0.177 | 0.028 | 0.108 | 0.246 | 0.092 | 0.265 |
| Kvasir-SEG-trained | precision | higher | 0.115 | 0.007 | 0.098 | 0.132 | 0.045 | 0.188 |
| Kvasir-SEG-trained | recall | higher | 0.147 | 0.067 | -0.020 | 0.313 | 0.055 | 0.244 |
| Kvasir-SEG-trained | hd95 | lower | -32.907 | 6.060 | -47.960 | -17.854 | -51.077 | -15.634 |
| Kvasir-SEG-trained | boundary_iou | higher | 0.102 | 0.010 | 0.076 | 0.128 | 0.047 | 0.158 |
| Kvasir-SEG-trained | miss | lower | -0.057 | 0.036 | -0.147 | 0.034 | -0.097 | -0.024 |
| CVC-ClinicDB-trained (preliminary (E10)) | dice | higher | -0.019 | 0.029 | -0.091 | 0.054 | -0.241 | 0.181 |
| CVC-ClinicDB-trained (preliminary (E10)) | iou | higher | 0.011 | 0.026 | -0.054 | 0.077 | -0.168 | 0.185 |
| CVC-ClinicDB-trained (preliminary (E10)) | precision | higher | 0.137 | 0.063 | -0.020 | 0.294 | -0.131 | 0.361 |
| CVC-ClinicDB-trained (preliminary (E10)) | recall | higher | -0.370 | 0.036 | -0.458 | -0.281 | -0.603 | -0.178 |
| CVC-ClinicDB-trained (preliminary (E10)) | hd95 | lower | 22.974 | 13.485 | -10.525 | 56.473 | -42.682 | 127.741 |
| CVC-ClinicDB-trained (preliminary (E10)) | boundary_iou | higher | 0.016 | 0.011 | -0.011 | 0.043 | -0.028 | 0.042 |
| CVC-ClinicDB-trained (preliminary (E10)) | miss | lower | 0.211 | 0.022 | 0.158 | 0.265 | 0.020 | 0.578 |

### Decision rule (Dice)
| direction | delta | mean_gap | ci_t_low | ci_t_high | decision_t_ci | ci_boot_low | ci_boot_high | decision_bootstrap_ci |
|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | 0.050 | 0.167 | 0.079 | 0.255 | meaningful drop | 0.087 | 0.251 | meaningful drop |
| Kvasir-SEG-trained | 0.030 | 0.167 | 0.079 | 0.255 | meaningful drop | 0.087 | 0.251 | meaningful drop |
| Kvasir-SEG-trained | 0.100 | 0.167 | 0.079 | 0.255 | inconclusive | 0.087 | 0.251 | inconclusive |
| CVC-ClinicDB-trained (preliminary (E10)) | 0.050 | -0.019 | -0.091 | 0.054 | inconclusive | -0.241 | 0.181 | inconclusive |
| CVC-ClinicDB-trained (preliminary (E10)) | 0.030 | -0.019 | -0.091 | 0.054 | inconclusive | -0.241 | 0.181 | inconclusive |
| CVC-ClinicDB-trained (preliminary (E10)) | 0.100 | -0.019 | -0.091 | 0.054 | equivalence | -0.241 | 0.181 | inconclusive |

### Spearman correlations (descriptive)
| trained_on | evaluated_on | variable | n_images | spearman_rho | p_value | status |
|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | CVC-ClinicDB | gt_area_fraction | 612 | 0.134 | 0.001 | nan |
| Kvasir-SEG-trained | CVC-ClinicDB | brightness_gray | 612 | 0.133 | 0.001 | nan |
| CVC-ClinicDB-trained | Kvasir-SEG | gt_area_fraction | 1000 | 0.798 | 0.000 | preliminary (E10) |
| CVC-ClinicDB-trained | Kvasir-SEG | brightness_gray | 1000 | -0.255 | 0.000 | preliminary (E10) |

### Error analysis
{
  "dice_threshold": 0.5,
  "evaluation_set": "cvc_external",
  "frames_below_threshold": 165,
  "model": "kvasir-trained",
  "reference_frames": 50,
  "reference_frames_drawn_from": "frames with seed-averaged Dice >= 0.5 (447 available)",
  "sampled_low_dice_frames": 100,
  "sampling_seed": 42,
  "size_bins": "small < 0.0363 (Q1 of GT area fraction at 256 px), large > 0.1272 (Q3), medium otherwise",
  "total_frames": 612
}

## Caveats
- Only 3 seeds: the t multiplier is 4.303, so intervals are wide.
- The CVC test set has 91 frames from 5 video sequences; the bootstrap CI for CVC resamples these 5 sequences.
- CVC-ClinicDB is split by video sequence, not by patient (see `outputs/audit_report.md`).
- An effect measured here is the combined effect of everything that differs between the datasets; it is not attributed to a single factor.
