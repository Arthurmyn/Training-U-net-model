# Error categories

Assign **one** primary category to every frame (columns `stem`, `category`, `notes` in your sheet). Use the number 1-7.
Look only at the figure (image, ground truth, one prediction per seed); the sheets contain no Dice values on purpose.
Work **independently**: do not discuss with the other annotator until both sheets are finished.

| # | Category | Use it when |
|---|---|---|
| 1 | Missed small polyp | A small ground-truth polyp is present and the model predicts nothing on it (or only something elsewhere). |
| 2 | False detection of folds or reflections | The prediction lies mainly on mucosal folds, ridges or specular highlights, not on the polyp. |
| 3 | Incomplete segmentation | The model finds the polyp but covers only part of it; a clear piece of the polyp is left out. |
| 4 | Inaccurate boundary | The right object is found with a roughly correct shape, but the outline is shifted, too wide or too tight. |
| 5 | Complex background | Debris, bubbles, instruments, blood, bile or strong texture around the polyp seem to mislead the model. |
| 6 | Low contrast | The polyp has almost the same colour or texture as its surroundings, or the frame is dark or blurred. |
| 7 | Several objects in the frame | More than one polyp or polyp-like structure; predictions and ground truth disagree about which one counts. |

Rules for hard cases
- If several categories fit, choose the one that explains the **largest part** of the error and mention the others in `notes`.
- Judge what you see in the predictions of all seeds together; if they differ a lot between seeds, describe the most frequent failure.
- Use exactly one of the seven categories for every frame; if none fits well, choose the closest one and explain in `notes`.
- Disagreements are resolved later in the `resolved_category` column of `disagreements.csv`, together with the other annotator.
