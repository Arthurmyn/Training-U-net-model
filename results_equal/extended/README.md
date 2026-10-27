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

**Check A** (recomputed Dice reproduces `runs/*/per_image_dice.csv`; required: per-evaluation-set mean absolute difference < 0.002 and at least 99% of images with a difference < 0.01): **PASSED** (unmatched rows: 0).

| evaluation set | images | mean abs diff | max abs diff | share of images with diff < 0.01 |
|---|---|---|---|---|
| cvc_external | 3060 | 0.00001 | 0.00189 | 1.00000 |
| cvc_test | 455 | 0.00001 | 0.00164 | 1.00000 |
| cvc_val | 460 | 0.00002 | 0.00134 | 1.00000 |
| kvasir_external | 5000 | 0.00001 | 0.00112 | 1.00000 |
| kvasir_test | 500 | 0.00001 | 0.00021 | 1.00000 |
| kvasir_val | 500 | 0.00001 | 0.00051 | 1.00000 |

**Check B** (IoU equals Dice / (2 - Dice) within 1e-4): **PASSED**, maximum deviation 2.22e-16.

## Numbers for the report

Runs completed: **10 of 10 planned** by the protocol (2 training datasets x 5 seeds). Evaluations completed: **30 in total (10 validation + 10 own-test + 10 external); 20 of 20 planned counting own-test and external evaluations.**
This folder contains 5 seeds per training dataset.

### Mean +/- SD over the 5 seeds
| evaluation set | images | dice | iou | precision | recall | hd95 | boundary_iou | miss |
|---|---|---|---|---|---|---|---|---|
| cvc_test (preliminary (E10)) | 91 | 0.348 +/- 0.047 | 0.261 +/- 0.045 | 0.484 +/- 0.049 | 0.353 +/- 0.050 | 126.58 +/- 13.49 | 0.072 +/- 0.020 | 0.180 +/- 0.052 |
| cvc_val (preliminary (E10)) | 92 | 0.473 +/- 0.012 | 0.359 +/- 0.015 | 0.538 +/- 0.051 | 0.581 +/- 0.074 | 91.15 +/- 16.42 | 0.097 +/- 0.014 | 0.096 +/- 0.039 |
| kvasir_external (preliminary (E10)) | 1000 | 0.413 +/- 0.018 | 0.291 +/- 0.016 | 0.352 +/- 0.003 | 0.762 +/- 0.041 | 109.94 +/- 1.66 | 0.065 +/- 0.012 | 0.001 +/- 0.001 |
| cvc_external | 612 | 0.590 +/- 0.054 | 0.498 +/- 0.063 | 0.667 +/- 0.085 | 0.639 +/- 0.028 | 83.20 +/- 6.52 | 0.218 +/- 0.053 | 0.091 +/- 0.026 |
| kvasir_test | 100 | 0.757 +/- 0.036 | 0.656 +/- 0.040 | 0.780 +/- 0.035 | 0.806 +/- 0.029 | 54.21 +/- 5.76 | 0.289 +/- 0.041 | 0.020 +/- 0.010 |
| kvasir_val | 100 | 0.786 +/- 0.053 | 0.694 +/- 0.061 | 0.836 +/- 0.058 | 0.794 +/- 0.035 | 52.12 +/- 7.88 | 0.331 +/- 0.058 | 0.022 +/- 0.004 |

HD95 in pixels (lower is better); miss is the share of images with an empty prediction (lower is better); all other metrics: higher is better.

### Gap = own test minus external (per seed, then summarised)
Primary CI: t-based, df = 4 (t = 2.776). Secondary CI: cluster bootstrap.

| direction | metric | better | mean_gap | sd_gap | ci_t_low | ci_t_high | ci_boot_low | ci_boot_high |
|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | dice | higher | 0.166 | 0.018 | 0.143 | 0.189 | 0.084 | 0.252 |
| Kvasir-SEG-trained | iou | higher | 0.157 | 0.023 | 0.129 | 0.186 | 0.074 | 0.243 |
| Kvasir-SEG-trained | precision | higher | 0.113 | 0.054 | 0.046 | 0.181 | 0.041 | 0.188 |
| Kvasir-SEG-trained | recall | higher | 0.167 | 0.018 | 0.144 | 0.190 | 0.070 | 0.269 |
| Kvasir-SEG-trained | hd95 | lower | -28.997 | 3.187 | -32.953 | -25.040 | -46.951 | -11.281 |
| Kvasir-SEG-trained | boundary_iou | higher | 0.070 | 0.014 | 0.053 | 0.088 | 0.025 | 0.116 |
| Kvasir-SEG-trained | miss | lower | -0.071 | 0.024 | -0.101 | -0.041 | -0.112 | -0.032 |
| CVC-ClinicDB-trained (preliminary (E10)) | dice | higher | -0.065 | 0.030 | -0.102 | -0.029 | -0.256 | 0.131 |
| CVC-ClinicDB-trained (preliminary (E10)) | iou | higher | -0.030 | 0.030 | -0.067 | 0.007 | -0.185 | 0.143 |
| CVC-ClinicDB-trained (preliminary (E10)) | precision | higher | 0.131 | 0.048 | 0.072 | 0.190 | -0.093 | 0.331 |
| CVC-ClinicDB-trained (preliminary (E10)) | recall | higher | -0.409 | 0.026 | -0.441 | -0.377 | -0.603 | -0.228 |
| CVC-ClinicDB-trained (preliminary (E10)) | hd95 | lower | 16.641 | 12.567 | 1.037 | 32.245 | -32.416 | 86.225 |
| CVC-ClinicDB-trained (preliminary (E10)) | boundary_iou | higher | 0.007 | 0.009 | -0.004 | 0.017 | -0.032 | 0.037 |
| CVC-ClinicDB-trained (preliminary (E10)) | miss | lower | 0.180 | 0.052 | 0.115 | 0.244 | 0.042 | 0.417 |

### Decision rule (Dice)
| direction | delta | mean_gap | ci_t_low | ci_t_high | decision_t_ci | ci_boot_low | ci_boot_high | decision_bootstrap_ci |
|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | 0.050 | 0.166 | 0.143 | 0.189 | meaningful drop | 0.084 | 0.252 | meaningful drop |
| Kvasir-SEG-trained | 0.030 | 0.166 | 0.143 | 0.189 | meaningful drop | 0.084 | 0.252 | meaningful drop |
| Kvasir-SEG-trained | 0.100 | 0.166 | 0.143 | 0.189 | meaningful drop | 0.084 | 0.252 | inconclusive |
| CVC-ClinicDB-trained (preliminary (E10)) | 0.050 | -0.065 | -0.102 | -0.029 | inconclusive | -0.256 | 0.131 | inconclusive |
| CVC-ClinicDB-trained (preliminary (E10)) | 0.030 | -0.065 | -0.102 | -0.029 | inconclusive | -0.256 | 0.131 | inconclusive |
| CVC-ClinicDB-trained (preliminary (E10)) | 0.100 | -0.065 | -0.102 | -0.029 | inconclusive | -0.256 | 0.131 | inconclusive |

### Spearman correlations (descriptive)
| trained_on | evaluated_on | variable | n_images | spearman_rho | p_value | status |
|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | CVC-ClinicDB | gt_area_fraction | 612 | 0.213 | 0.000 | nan |
| Kvasir-SEG-trained | CVC-ClinicDB | brightness_gray | 612 | 0.195 | 0.000 | nan |
| CVC-ClinicDB-trained | Kvasir-SEG | gt_area_fraction | 1000 | 0.785 | 0.000 | preliminary (E10) |
| CVC-ClinicDB-trained | Kvasir-SEG | brightness_gray | 1000 | -0.268 | 0.000 | preliminary (E10) |

### Error analysis
not generated yet (python -m train.extended errors)

## Caveats
- 5 seeds per dataset: the t multiplier is 2.776.
- The CVC test set has 91 frames from 5 video sequences; the bootstrap CI for CVC resamples these 5 sequences.
- CVC-ClinicDB is split by video sequence, not by patient (see `outputs/audit_report.md`).
- An effect measured here is the combined effect of everything that differs between the datasets; it is not attributed to a single factor.
