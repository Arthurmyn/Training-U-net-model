# Results

Dice per image at probability threshold 0.5, computed on 256x256 preprocessed masks. Confidence intervals: 10000 bootstrap samples, 95%, whole video sequences resampled where sequence IDs exist (frames of one sequence are not independent), otherwise images. Seeds are averaged per image first; the spread between seeds is the last column of table 1.

## 1. Dice by condition

| trained on | evaluated on | condition | eval_set | images | clusters | seeds | mean Dice | CI95 low | CI95 high | SD across seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| CVC-ClinicDB | Kvasir-SEG | external (whole other dataset) | kvasir_external | 1000 | 1000 | 5 | 0.413 | 0.399 | 0.427 | 0.018 |
| CVC-ClinicDB | CVC-ClinicDB | in-domain test | cvc_test | 91 | 5 | 5 | 0.348 | 0.155 | 0.545 | 0.047 |
| CVC-ClinicDB | CVC-ClinicDB | in-domain test | cvc_val | 92 | 5 | 5 | 0.473 | 0.350 | 0.597 | 0.012 |
| Kvasir-SEG | CVC-ClinicDB | external (whole other dataset) | cvc_external | 612 | 29 | 5 | 0.590 | 0.513 | 0.664 | 0.054 |
| Kvasir-SEG | Kvasir-SEG | in-domain test | kvasir_test | 100 | 100 | 5 | 0.757 | 0.716 | 0.795 | 0.036 |
| Kvasir-SEG | Kvasir-SEG | in-domain test | kvasir_val | 100 | 100 | 5 | 0.786 | 0.746 | 0.823 | 0.053 |

## 2. Drop in Dice when a model is applied to the other dataset

Drop = in-domain test Dice minus Dice on the whole other dataset (different images, so the two sets are resampled independently).

| trained on | in-domain Dice | external Dice | drop (in-domain - external) | CI95 low | CI95 high | reading at 0.05 | reading at 0.03 | reading at 0.1 |
|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 0.757 | 0.590 | 0.166 | 0.083 | 0.252 | drop > 0.05 (CI entirely above) | drop > 0.03 (CI entirely above) | inconclusive for 0.1 |
| CVC-ClinicDB | 0.348 | 0.413 | -0.065 | -0.252 | 0.132 | inconclusive for 0.05 | inconclusive for 0.03 | inconclusive for 0.1 |

## 3. Same target images, different training dataset (paired)

Dice of the model trained on the target dataset versus the model trained on the other dataset, on the same test images of the target dataset.

| target (test images) | images | clusters | Dice, trained on target | Dice, trained on other | difference (own - other) | CI95 low | CI95 high | reading at 0.05 | reading at 0.03 | reading at 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 100 | 100 | 0.757 | 0.416 | 0.341 | 0.293 | 0.388 | drop > 0.05 (CI entirely above) | drop > 0.03 (CI entirely above) | drop > 0.1 (CI entirely above) |
| CVC-ClinicDB | 91 | 5 | 0.348 | 0.448 | -0.100 | -0.366 | 0.097 | inconclusive for 0.05 | inconclusive for 0.03 | drop < 0.1 (CI entirely below) |

Reading of an interval against a practical threshold t (main 0.05, sensitivity 0.03, 0.1): 'drop > t' if the whole interval lies above t, 'drop < t' if it lies below, otherwise inconclusive. These are decision rules chosen for this pipeline; check them against the hypotheses in the methodology.

## What these numbers do not show

- **Statistical uncertainty** is what the intervals describe: how much the result would vary with other images (and sequences).
- **Practical significance** is the comparison with the threshold t, decided before looking at results.
- **Cause** is not established. The experiment measures the combined effect of moving between datasets (brightness, resolution, endoscope, annotation style, patient population, number of training images). It cannot attribute a Dice drop to any single factor.
- CVC-ClinicDB splits are by video sequence, not by patient, so some patient-level overlap between train and test cannot be excluded (see outputs/audit_report.md).

Figure: `fig_dice_distributions.png`. Tables: `tables/`.
