# Results

Dice per image at probability threshold 0.5, computed on 256x256 preprocessed masks. Confidence intervals: 10000 bootstrap samples, 95%, whole video sequences resampled where sequence IDs exist (frames of one sequence are not independent), otherwise images. Seeds are averaged per image first; the spread between seeds is the last column of table 1.

## 1. Dice by condition

| trained on | evaluated on | condition | eval_set | images | clusters | seeds | mean Dice | CI95 low | CI95 high | SD across seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| CVC-ClinicDB | Kvasir-SEG | external (whole other dataset) | kvasir_external | 1000 | 1000 | 3 | 0.397 | 0.383 | 0.411 | 0.007 |
| CVC-ClinicDB | CVC-ClinicDB | in-domain test | cvc_test | 91 | 5 | 3 | 0.378 | 0.156 | 0.575 | 0.036 |
| CVC-ClinicDB | CVC-ClinicDB | in-domain test | cvc_val | 92 | 5 | 3 | 0.467 | 0.329 | 0.596 | 0.046 |
| Kvasir-SEG | CVC-ClinicDB | external (whole other dataset) | cvc_external | 612 | 29 | 3 | 0.655 | 0.578 | 0.724 | 0.036 |
| Kvasir-SEG | Kvasir-SEG | in-domain test | kvasir_test | 100 | 100 | 3 | 0.822 | 0.781 | 0.858 | 0.003 |
| Kvasir-SEG | Kvasir-SEG | in-domain test | kvasir_val | 100 | 100 | 3 | 0.854 | 0.820 | 0.885 | 0.010 |

## 2. Drop in Dice when a model is applied to the other dataset

Drop = in-domain test Dice minus Dice on the whole other dataset (different images, so the two sets are resampled independently).

| trained on | in-domain Dice | external Dice | drop (in-domain - external) | CI95 low | CI95 high | reading at 0.05 | reading at 0.03 | reading at 0.1 |
|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 0.822 | 0.655 | 0.167 | 0.087 | 0.252 | drop > 0.05 (CI entirely above) | drop > 0.03 (CI entirely above) | inconclusive for 0.1 |
| CVC-ClinicDB | 0.378 | 0.397 | -0.019 | -0.237 | 0.182 | inconclusive for 0.05 | inconclusive for 0.03 | inconclusive for 0.1 |

## 3. Same target images, different training dataset (paired)

Dice of the model trained on the target dataset versus the model trained on the other dataset, on the same test images of the target dataset.

| target (test images) | images | clusters | Dice, trained on target | Dice, trained on other | difference (own - other) | CI95 low | CI95 high | reading at 0.05 | reading at 0.03 | reading at 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 100 | 100 | 0.822 | 0.397 | 0.425 | 0.374 | 0.474 | drop > 0.05 (CI entirely above) | drop > 0.03 (CI entirely above) | drop > 0.1 (CI entirely above) |
| CVC-ClinicDB | 91 | 5 | 0.378 | 0.515 | -0.136 | -0.413 | 0.021 | drop < 0.05 (CI entirely below) | drop < 0.03 (CI entirely below) | drop < 0.1 (CI entirely below) |

Reading of an interval against a practical threshold t (main 0.05, sensitivity 0.03, 0.1): 'drop > t' if the whole interval lies above t, 'drop < t' if it lies below, otherwise inconclusive. These are decision rules chosen for this pipeline; check them against the hypotheses in the methodology.

## What these numbers do not show

- **Statistical uncertainty** is what the intervals describe: how much the result would vary with other images (and sequences).
- **Practical significance** is the comparison with the threshold t, decided before looking at results.
- **Cause** is not established. The experiment measures the combined effect of moving between datasets (brightness, resolution, endoscope, annotation style, patient population, number of training images). It cannot attribute a Dice drop to any single factor.
- CVC-ClinicDB splits are by video sequence, not by patient, so some patient-level overlap between train and test cannot be excluded (see outputs/audit_report.md).

Figure: `fig_dice_distributions.png`. Tables: `tables/`.
