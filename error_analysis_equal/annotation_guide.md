# Annotation guide

Two annotators label the same frames independently.

1. Open the annotator's own sheet (`annotator_A.csv` or `annotator_B.csv`) and the panels in `frames/`.
2. Look at the panel `frames/<stem>.png` for each row. Dice values are not available to the annotators.
3. Enter one category number (1-7) in the `category` column. One category per frame; the closest one if none fits well. Optional remarks go to `notes`.
4. Do not discuss frames with the other annotator, and do not look at the other sheet, until both sheets are complete.
5. Do not edit the `stem` column or the order of the rows.

## Categories

1. Missed small polyp
2. False detection of folds or reflections
3. Incomplete segmentation
4. Inaccurate boundary
5. Complex background
6. Low contrast
7. Several objects in the frame

The full description of each category is in `CATEGORIES.md`.
