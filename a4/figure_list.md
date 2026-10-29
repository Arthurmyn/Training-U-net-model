# Figure list

Every PNG of `final_results/figures/` and `all_figures/`. A figure is marked as showing dataset frames when it contains image content of the datasets (frames with overlays). 

- Figures in `final_results/figures/`: 13
- Figures in `all_figures/`: 26
- Total: 39
- Types: diagram 2, graph 33, image panels 4
- Showing dataset frames: 4

| path | title | description | type | shows dataset frames | pixels |
|---|---|---|---|---|---|
| final_results/figures/fig01_study_design.png | Study design | datasets, splits, models and the four conditions E1-E4 | diagram | no | 2400 x 1120 |
| final_results/figures/fig02_dataset_differences.png | Dataset differences | polyp area, mean brightness and image size of both datasets | graph | no | 3000 x 880 |
| final_results/figures/fig03_dice_by_condition.png | Dice by condition | per-image Dice distribution and seed means for E1-E4 | graph | no | 1900 x 1080 |
| final_results/figures/fig04_gap_per_seed_dice.png | Generalization gap per seed | gap G per seed with the t-based 95% interval and the delta thresholds | graph | no | 1280 x 1000 |
| final_results/figures/fig05_all_metrics.png | All metrics | Dice, IoU, precision, recall, HD95, Boundary IoU per condition (seeds and 95% interval) | graph | no | 2800 x 1640 |
| final_results/figures/fig06_gap_all_metrics.png | Gap for all metrics | gap of every metric with its interval | graph | no | 3000 x 1360 |
| final_results/figures/fig07_dice_by_polyp_size.png | Dice by polyp size | small / medium / large polyps (quartiles of the true area) | graph | no | 2100 x 1120 |
| final_results/figures/fig08_examples_E2.png | Examples E2 | best, median, worst frame of the external set | image panels | yes | 2200 x 1720 |
| final_results/figures/fig08_examples_E4.png | Examples E4 | best, median, worst frame of the external set | image panels | yes | 2200 x 1720 |
| final_results/figures/fig09_equal_size_comparison.png | Comparison with the earlier preliminary run set | Dice of the earlier run set (800 vs 429 training images) next to the final experiment | graph | no | 2300 x 1040 |
| final_results/figures/fig10_metric_distributions.png | Metric distributions | per-image Dice, IoU, HD95 and Boundary IoU | graph | no | 2500 x 1720 |
| final_results/figures/figA1_training_curves.png | Training curves | loss and Dice per epoch, one line per seed | graph | no | 2240 x 1440 |
| final_results/figures/figA2_delta_sensitivity.png | Sensitivity to the threshold | decision of the rule for delta = 0.05, 0.03, 0.10 | graph | no | 3120 x 720 |
| all_figures/01_data/D1_study_design.png | Study design | datasets, splits, models and the four conditions E1-E4 | diagram | no | 2400 x 1120 |
| all_figures/01_data/D2_dataset_differences.png | Dataset differences | polyp area, mean brightness and image size of both datasets | graph | no | 3000 x 880 |
| all_figures/01_data/D3_colour_polyps_aspect.png | Colour, polyps per mask, aspect ratio | colour channels, number of polyps per mask and image aspect ratio | graph | no | 3000 x 919 |
| all_figures/01_data/D4_split_overview.png | Splits | train/val/test sizes and frames per video sequence coloured by part | graph | no | 2600 x 919 |
| all_figures/01_data/D5_near_duplicates.png | Near-duplicate candidates | image pairs per perceptual-hash distance, within and between the datasets | graph | no | 1680 x 919 |
| all_figures/02_training/T1_training_curves.png | Training curves | loss and Dice per epoch, one line per seed | graph | no | 2240 x 1440 |
| all_figures/02_training/T2_validation_and_stopping.png | Validation Dice and stopping | validation Dice of every run, selected epoch and stopping epoch | graph | no | 2800 x 1000 |
| all_figures/02_training/T3_train_validation_gap.png | Train vs validation gap | train Dice minus validation Dice over the epochs | graph | no | 1680 x 960 |
| all_figures/03_results/R1_dice_by_condition.png | Dice by condition | per-image Dice distribution and seed means for E1-E4 | graph | no | 1900 x 1080 |
| all_figures/03_results/R2_gap_per_seed.png | Generalization gap per seed | gap G per seed with the t-based 95% interval and the delta thresholds | graph | no | 1280 x 1000 |
| all_figures/03_results/R3_all_metrics.png | All metrics | Dice, IoU, precision, recall, HD95, Boundary IoU per condition (seeds and 95% interval) | graph | no | 2800 x 1639 |
| all_figures/03_results/R4_gap_all_metrics.png | Gap for all metrics | gap of every metric with its interval | graph | no | 3000 x 1360 |
| all_figures/03_results/R5_metric_distributions.png | Metric distributions | per-image Dice, IoU, HD95 and Boundary IoU | graph | no | 2500 x 1720 |
| all_figures/03_results/R6_delta_sensitivity.png | Sensitivity to the threshold | decision of the rule for delta = 0.05, 0.03, 0.10 | graph | no | 3120 x 720 |
| all_figures/03_results/R7_dice_ecdf.png | Cumulative distribution of Dice | share of images below each Dice value for E1-E4 | graph | no | 1639 x 1040 |
| all_figures/03_results/R8_paired_same_images.png | Same images, two models | the same test images segmented by both models (paired) | graph | no | 2240 x 1080 |
| all_figures/03_results/R9_precision_recall.png | Precision vs recall | over- vs under-segmentation per condition and seed | graph | no | 1440 x 1240 |
| all_figures/04_errors/F1_dice_by_polyp_size.png | Dice by polyp size | small / medium / large polyps (quartiles of the true area) | graph | no | 2100 x 1120 |
| all_figures/04_errors/F2_dice_vs_polyp_size.png | Dice vs polyp size (per image) | continuous dependence of Dice on the true polyp area | graph | no | 2200 x 1680 |
| all_figures/04_errors/F3_predicted_vs_true_area.png | Predicted vs true area | calibration of the predicted polyp area; empty predictions | graph | no | 2200 x 1720 |
| all_figures/04_errors/F4_dice_vs_brightness.png | Dice vs brightness | descriptive look at the brightness shift (Spearman rho) | graph | no | 2200 x 1680 |
| all_figures/04_errors/F5_dice_by_sequence.png | Dice per video sequence | dependence of the result on the evaluated sequences | graph | no | 2600 x 1560 |
| all_figures/04_errors/F6_examples_E2.png | Examples E2 | best, median, worst frame of the external set | image panels | yes | 2200 x 1720 |
| all_figures/04_errors/F7_examples_E4.png | Examples E4 | best, median, worst frame of the external set | image panels | yes | 2200 x 1720 |
| all_figures/05_robustness/B1_original_vs_256.png | Original resolution vs 256 x 256 | per-image Dice at the original size against the 256 x 256 Dice | graph | no | 2200 x 960 |
| all_figures/05_robustness/B2_gap_by_seed_and_machine.png | Gap per seed and machine | gap of each seed with the GPU that trained it | graph | no | 1480 x 1000 |
