# All figures of the experiment

Built by `python -m train.figure_gallery`; the folder is rebuilt from scratch every time. Conditions: E1 = first dataset on its own test set, E2 = first dataset on the other dataset, E3 = second dataset on its own test set, E4 = second dataset on the first dataset. Figures that show frames of the datasets (E6, E7) are for research and educational use only.

## Data and design

| File | Figure | What it shows | Data |
|---|---|---|---|
| [`D1_study_design.png`](01_data/D1_study_design.png) | Study design | datasets, splits, models and the four conditions E1-E4 | splits, run.json |
| [`D2_dataset_differences.png`](01_data/D2_dataset_differences.png) | Dataset differences | polyp area, mean brightness and image size of both datasets | outputs/per_image_stats_*.csv |
| [`D3_colour_polyps_aspect.png`](01_data/D3_colour_polyps_aspect.png) | Colour, polyps per mask, aspect ratio | colour channels, number of polyps per mask and image aspect ratio | outputs/per_image_stats_*.csv |
| [`D4_split_overview.png`](01_data/D4_split_overview.png) | Splits | train/val/test sizes and frames per video sequence coloured by part | outputs/splits/*.csv |
| [`D5_near_duplicates.png`](01_data/D5_near_duplicates.png) | Near-duplicate candidates | image pairs per perceptual-hash distance, within and between the datasets | outputs/duplicate_pairs.csv |

## Training

| File | Figure | What it shows | Data |
|---|---|---|---|
| [`T1_training_curves.png`](02_training/T1_training_curves.png) | Training curves | loss and Dice per epoch, one line per seed | runs_equal/*/metrics.csv |
| [`T2_validation_and_stopping.png`](02_training/T2_validation_and_stopping.png) | Validation Dice and stopping | validation Dice of every run, selected epoch and stopping epoch | runs_equal/*/metrics.csv, run.json |
| [`T3_train_validation_gap.png`](02_training/T3_train_validation_gap.png) | Train vs validation gap | train Dice minus validation Dice over the epochs | runs_equal/*/metrics.csv |

## Results

| File | Figure | What it shows | Data |
|---|---|---|---|
| [`R1_dice_by_condition.png`](03_results/R1_dice_by_condition.png) | Dice by condition | per-image Dice distribution and seed means for E1-E4 | per_image_metrics_extended.csv |
| [`R2_gap_per_seed.png`](03_results/R2_gap_per_seed.png) | Generalization gap per seed | gap G per seed with the t-based 95% interval and the delta thresholds | per_image_metrics_extended.csv |
| [`R3_all_metrics.png`](03_results/R3_all_metrics.png) | All metrics | Dice, IoU, precision, recall, HD95, Boundary IoU per condition (seeds and 95% interval) | per_image_metrics_extended.csv |
| [`R4_gap_all_metrics.png`](03_results/R4_gap_all_metrics.png) | Gap for all metrics | gap of every metric with its interval | per_image_metrics_extended.csv |
| [`R5_metric_distributions.png`](03_results/R5_metric_distributions.png) | Metric distributions | per-image Dice, IoU, HD95 and Boundary IoU | per_image_metrics_extended.csv |
| [`R6_delta_sensitivity.png`](03_results/R6_delta_sensitivity.png) | Sensitivity to the threshold | decision of the rule for delta = 0.05, 0.03, 0.10 | gap_summary.csv |
| [`R7_dice_ecdf.png`](03_results/R7_dice_ecdf.png) | Cumulative distribution of Dice | share of images below each Dice value for E1-E4 | per_image_metrics_extended.csv |
| [`R8_paired_same_images.png`](03_results/R8_paired_same_images.png) | Same images, two models | the same test images segmented by both models (paired) | per_image_metrics_extended.csv |
| [`R9_precision_recall.png`](03_results/R9_precision_recall.png) | Precision vs recall | over- vs under-segmentation per condition and seed | per_image_metrics_extended.csv |

## Errors and their causes

| File | Figure | What it shows | Data |
|---|---|---|---|
| [`F1_dice_by_polyp_size.png`](04_errors/F1_dice_by_polyp_size.png) | Dice by polyp size | small / medium / large polyps (quartiles of the true area) | per_image_metrics_extended.csv |
| [`F2_dice_vs_polyp_size.png`](04_errors/F2_dice_vs_polyp_size.png) | Dice vs polyp size (per image) | continuous dependence of Dice on the true polyp area | per_image_metrics_extended.csv |
| [`F3_predicted_vs_true_area.png`](04_errors/F3_predicted_vs_true_area.png) | Predicted vs true area | calibration of the predicted polyp area; empty predictions | per_image_metrics_extended.csv |
| [`F4_dice_vs_brightness.png`](04_errors/F4_dice_vs_brightness.png) | Dice vs brightness | descriptive look at the brightness shift (Spearman rho) | per_image_metrics_extended.csv, outputs/per_image_stats_*.csv |
| [`F5_dice_by_sequence.png`](04_errors/F5_dice_by_sequence.png) | Dice per video sequence | dependence of the result on the evaluated sequences | per_image_metrics_extended.csv |
| [`F6_examples_E2.png`](04_errors/F6_examples_E2.png) | Examples E2 | best, median, worst frame of the external set (shows dataset frames) | weights, data |
| [`F7_examples_E4.png`](04_errors/F7_examples_E4.png) | Examples E4 | best, median, worst frame of the external set (shows dataset frames) | weights, data |

## Robustness checks

| File | Figure | What it shows | Data |
|---|---|---|---|
| [`B1_original_vs_256.png`](05_robustness/B1_original_vs_256.png) | Original resolution vs 256 x 256 | per-image Dice at the original size against the 256 x 256 Dice | results_equal/original_resolution |
| [`B2_gap_by_seed_and_machine.png`](05_robustness/B2_gap_by_seed_and_machine.png) | Gap per seed and machine | gap of each seed with the GPU that trained it | gap_by_seed.csv, run.json |

