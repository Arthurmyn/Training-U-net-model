# Metric definitions

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
| Boundary IoU | Band B_d(M) = pixels of M whose distance to the nearest background pixel is <= d, computed on M padded with a 1-pixel zero border, d = 5 pixels (`extended.boundary_iou_d` in `train_config.yaml`). BIoU = \|B_d(pred) AND B_d(gt)\| / \|B_d(pred) OR B_d(gt)\|. Empty prediction: 0. For masks narrower than 2d the band equals the mask, so BIoU equals IoU. |
| miss | 1 if the prediction contains no polyp pixel, else 0 (its mean over images is the miss rate). |

## Statistics

- **Seed-level mean** M_s: mean of a metric over the images of one evaluation set, for seed s. "Mean over seeds" and the sample SD (ddof = 1) are taken over the 3 values M_s.
- **Gap per seed** G_s = M_s(own test) - M_s(external): Kvasir-trained: kvasir_test minus cvc_external; CVC-trained: cvc_test minus kvasir_external. For HD95 and miss lower is better, so a negative gap means the model does better on its own data; `better` states the direction in every table.
- **Primary CI:** mean(G_s) +/- t * SD(G_s) / sqrt(3), with t = t.ppf(0.975, 2) = 4.303 (df = 2).
- **Secondary CI:** cluster bootstrap (10 000 resamples, 95%, seed 42, `train/analyze.py`): images are averaged over seeds, whole video sequences are resampled for CVC frames, single images for Kvasir; own and external sets are resampled independently.
- **Decision rule (Dice only)** on a CI [L, U] and a practical threshold delta in {0.05 (main), 0.03, 0.10}: L > delta: *meaningful drop*; U < -delta: *meaningful increase*; [L, U] inside [-delta, delta]: *equivalence*; otherwise *inconclusive*.
- **Spearman correlations** (descriptive): seed-mean Dice per image on the external set against (a) the ground-truth area fraction at 256 px and (b) the mean grey brightness from `outputs/per_image_stats_*.csv`. p-values are descriptive: frames of one video sequence are not independent.
- **median_over_images** (in `metrics_by_condition.csv`) is the mean over seeds of the per-seed median over images; it is added because the mean HD95 is dominated by missed polyps (362.04 each).

## Preliminary results

CVC-trained models are under-trained (error E10 in `docs/errors_log.md`). Every table row and figure for them carries the label **preliminary (E10)**.
