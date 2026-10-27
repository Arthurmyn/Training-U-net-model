# Collected results

Everything in this folder was produced from `equal_size.yaml` by `python -m train.collect`. Nothing here is edited by hand.

- Runs: 10 (cvc_seed1, cvc_seed2, cvc_seed3, cvc_seed4, cvc_seed5, kvasir_seed1, kvasir_seed2, kvasir_seed3, kvasir_seed4, kvasir_seed5).
- Models: U-Net, 32 base channels, 256 x 256 input; metrics are computed on the 256 x 256 masks at probability threshold 0.5.

## Contents
| Folder or file | Content |
|---|---|
| `figures/` | figures for the paper: fig01 study design, fig02 dataset differences, fig03 Dice by condition, fig04 gap per seed, fig05 all metrics, fig06 gap for all metrics, fig07 Dice by polyp size, fig08 qualitative examples (not stored: they show dataset frames; build with `--with-frames`), fig09 equal-size comparison (if an earlier experiment exists), fig10 metric distributions, figA* appendix |
| `tables/` | `metrics_by_condition.csv` (mean and SD over seeds, every metric), `gap_by_seed.csv`, `gap_summary.csv` (t-based and bootstrap CI), `decision_rule.csv`, `spearman.csv`, `acceptance_checks.json`, `original_analysis/` |
| `per_image/` | one row per run, evaluation set and image with Dice, IoU, precision, recall, HD95, Boundary IoU, miss |
| `runs/` | per run: training log per epoch (`metrics.csv`), settings and environment (`run.json`), evaluation summary, per-image Dice |
| `config/` | the configuration files used, the fixed splits and the fixed training subset |
| `audit/` | dataset audit report |
| `METRICS.md` | definitions of all metrics and statistics |
| `MANIFEST.json` | list of all files with SHA-256 checksums, versions and the git commit |

## Key numbers
| condition | images | seeds | Dice mean | SD over seeds |
|---|---|---|---|---|
| E1  Kvasir-SEG → Kvasir-SEG (own test) | 100 | 5 | 0.757 | 0.036 |
| E2  Kvasir-SEG → CVC-ClinicDB (external) | 612 | 5 | 0.590 | 0.054 |
| E3  CVC-ClinicDB → CVC-ClinicDB (own test) preliminary (E10) | 91 | 5 | 0.348 | 0.047 |
| E4  CVC-ClinicDB → Kvasir-SEG (external) preliminary (E10) | 1000 | 5 | 0.413 | 0.018 |

Dice gap (own test minus external), t-based CI (primary) and cluster bootstrap CI:

| direction | mean_gap | ci_t_low | ci_t_high | ci_boot_low | ci_boot_high | status |
|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | 0.166 | 0.143 | 0.189 | 0.084 | 0.252 |  |
| CVC-ClinicDB-trained | -0.065 | -0.102 | -0.029 | -0.256 | 0.131 | preliminary (E10) |

Decision rule, main threshold:

| direction | delta | decision_t_ci | decision_bootstrap_ci |
|---|---|---|---|
| Kvasir-SEG-trained | 0.050 | meaningful drop | meaningful drop |
| CVC-ClinicDB-trained | 0.050 | inconclusive | inconclusive |

Dice gaps are reported with the t-based 95% CI over seeds as primary interval. An effect measured here is the combined effect of everything that differs between the datasets; it is not attributed to one factor.
