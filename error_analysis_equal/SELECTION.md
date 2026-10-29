# Selection of the frames for the manual annotation

Model: kvasir-trained (seeds 1-5, `runs_equal/`). Evaluation set: `cvc_external` (612 frames).

## Rule

1. Per frame, the Dice values of the five seeds are averaged.
2. Frames with seed-averaged Dice < 0.5 are candidates: 199 of 612.
3. If there are more candidates than 100, a random sample of that size is drawn without replacement (sampling seed 42).
4. 50 reference frames are drawn at random (same sampling seed) from the frames with Dice >= 0.5; frames with seed-averaged Dice >= 0.5 (413 available).
5. Polyp-size bins: small < 0.0363 (Q1 of GT area fraction at 256 px), large > 0.1272 (Q3), medium otherwise.

## Counts

- Frames in the annotation set: 150 (low Dice: 100, reference: 50).
- Video sequences represented: 29.

| group | small | medium | large |
|---|---|---|---|
| low_dice | 33 | 46 | 21 |
| reference | 10 | 32 | 8 |

## Files

- `frames/<stem>.png`: panel of one frame (image, ground truth, one prediction per seed). No Dice value is drawn.
- `annotator_A.csv`, `annotator_B.csv`: the same frames in a different shuffled order each; `category` and `notes` are empty.
- `CATEGORIES.md`, `annotation_guide.md`: category rules and procedure.
- `frames.csv`, `selection.json`, `failure_cases.md`: contain Dice values and are not for the annotators.
