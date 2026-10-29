# Failure cases

Selection rule: the 3 frames with the lowest seed-averaged Dice among the frames of the annotation set (`frames.csv`; ties are broken by the frame name in text order), then one frame for each polyp-size bin (the lower median by Dice of the bin, taken from a video sequence not used yet). Areas are fractions of the 256 x 256 image. Category and cause are left blank for the annotators.

## Frame 178

- Selected as: lowest Dice
- Sequence: 9; polyp-size bin: small; group: low_dice
- Mean Dice over seeds: 0.000
- Dice per seed: seed 1: 0.000, seed 2: 0.000, seed 3: 0.000, seed 4: 0.000, seed 5: 0.000
- Ground-truth area: 0.0039
- Predicted area per seed: seed 1: 0.0097, seed 2: 0.0000, seed 3: 0.0637, seed 4: 0.0000, seed 5: 0.0000
- Empty prediction per seed: seed 1: no, seed 2: yes, seed 3: no, seed 4: yes, seed 5: yes
- Panel: [`frames/178.png`](frames/178.png)
- Category: 
- Cause: 

## Frame 246

- Selected as: lowest Dice
- Sequence: 12; polyp-size bin: medium; group: low_dice
- Mean Dice over seeds: 0.000
- Dice per seed: seed 1: 0.000, seed 2: 0.000, seed 3: 0.000, seed 4: 0.000, seed 5: 0.000
- Ground-truth area: 0.1116
- Predicted area per seed: seed 1: 0.0000, seed 2: 0.1787, seed 3: 0.0000, seed 4: 0.0000, seed 5: 0.0000
- Empty prediction per seed: seed 1: yes, seed 2: no, seed 3: yes, seed 4: no, seed 5: yes
- Panel: [`frames/246.png`](frames/246.png)
- Category: 
- Cause: 

## Frame 354

- Selected as: lowest Dice
- Sequence: 17; polyp-size bin: medium; group: low_dice
- Mean Dice over seeds: 0.000
- Dice per seed: seed 1: 0.000, seed 2: 0.000, seed 3: 0.000, seed 4: 0.000, seed 5: 0.000
- Ground-truth area: 0.0695
- Predicted area per seed: seed 1: 0.0000, seed 2: 0.0000, seed 3: 0.0026, seed 4: 0.0000, seed 5: 0.0000
- Empty prediction per seed: seed 1: yes, seed 2: yes, seed 3: no, seed 4: yes, seed 5: yes
- Panel: [`frames/354.png`](frames/354.png)
- Category: 
- Cause: 

## Frame 74

- Selected as: median Dice of the small bin
- Sequence: 4; polyp-size bin: small; group: low_dice
- Mean Dice over seeds: 0.396
- Dice per seed: seed 1: 0.569, seed 2: 0.452, seed 3: 0.446, seed 4: 0.000, seed 5: 0.515
- Ground-truth area: 0.0249
- Predicted area per seed: seed 1: 0.0204, seed 2: 0.0420, seed 3: 0.0444, seed 4: 0.0000, seed 5: 0.0128
- Empty prediction per seed: seed 1: no, seed 2: no, seed 3: no, seed 4: yes, seed 5: no
- Panel: [`frames/74.png`](frames/74.png)
- Category: 
- Cause: 

## Frame 283

- Selected as: median Dice of the medium bin
- Sequence: 14; polyp-size bin: medium; group: low_dice
- Mean Dice over seeds: 0.435
- Dice per seed: seed 1: 0.934, seed 2: 0.283, seed 3: 0.000, seed 4: 0.957, seed 5: 0.000
- Ground-truth area: 0.0367
- Predicted area per seed: seed 1: 0.0330, seed 2: 0.1608, seed 3: 0.0000, seed 4: 0.0372, seed 5: 0.0000
- Empty prediction per seed: seed 1: no, seed 2: no, seed 3: yes, seed 4: no, seed 5: yes
- Panel: [`frames/283.png`](frames/283.png)
- Category: 
- Cause: 

## Frame 472

- Selected as: median Dice of the large bin
- Sequence: 23; polyp-size bin: large; group: low_dice
- Mean Dice over seeds: 0.222
- Dice per seed: seed 1: 0.000, seed 2: 0.047, seed 3: 0.567, seed 4: 0.155, seed 5: 0.340
- Ground-truth area: 0.2080
- Predicted area per seed: seed 1: 0.0000, seed 2: 0.0051, seed 3: 0.0857, seed 4: 0.0175, seed 5: 0.0429
- Empty prediction per seed: seed 1: yes, seed 2: no, seed 3: no, seed 4: no, seed 5: no
- Panel: [`frames/472.png`](frames/472.png)
- Category: 
- Cause: 
