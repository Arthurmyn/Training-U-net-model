# Dataset audit report

Seed 42; Python 3.9.6, numpy 1.26.4, pandas 2.3.3, scipy 1.13.1, Pillow 11.3.0, matplotlib 3.9.4. Masks are binarised with value > 127; a mask is 'near-full' above 90% polyp area; near-duplicates are images whose 64-bit perceptual hash differs in at most 4 bits.

## 1. Pair counts

| Dataset | Images | Masks | Matched pairs | Unmatched images | Unmatched masks | Expected | Status |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 1000 | 1000 | 1000 | 0 | 0 | 1000 | OK |
| CVC-ClinicDB | 612 | 612 | 612 | 0 | 0 | 612 | OK |

- **Kvasir-SEG:** 1000 matched image-mask pairs were found, as expected (1000); no unmatched files.
- **CVC-ClinicDB:** 612 matched image-mask pairs were found, as expected (612); no unmatched files.

## 2. Problems found

- Kvasir-SEG: 1000 of 1000 masks contain values other than 0/255 (up to 18 distinct values in one mask; compression or anti-aliasing artefacts). They are binarised with > 127; 1000 of 1000 binarised masks have exactly two classes.
- CVC-ClinicDB: 138 of 612 masks contain values other than 0/255 (up to 64 distinct values in one mask; compression or anti-aliasing artefacts). They are binarised with > 127; 612 of 612 binarised masks have exactly two classes.
- CVC-ClinicDB: the sequence metadata lists 29 video sequences, but 31 were expected.
- CVC-ClinicDB: the second (TIFF) copy of 612 images could not be decoded by Pillow, so the PNG/TIFF image cross-check was not possible (masks: 612/612 identical).
- 183 near-duplicate image pair(s) within CVC-ClinicDB (pHash Hamming <= 4) (181 of them within the same video sequence).
- 7 near-duplicate image pair(s) within Kvasir-SEG (pHash Hamming <= 4).
- Kvasir-SEG: 2 near-duplicate pair(s) end up in different splits.

## 3. Unreadable files

- **Kvasir-SEG:** all 1000 images and all masks opened and decoded completely.
- **CVC-ClinicDB:** all 612 images and all masks opened and decoded completely.

## 4. Image sizes

| Dataset | Distinct sizes | Width px | Height px | Channels | Image/mask size mismatches |
|---|---|---|---|---|---|
| Kvasir-SEG | 333 | 332-1920 (median 621) | 352-1072 (median 530) | 3 ch: 1000 | 0 |
| CVC-ClinicDB | 1 | 384-384 (median 384) | 288-288 (median 288) | 3 ch: 612 | 0 |

Most frequent sizes, Kvasir-SEG (full table: `image_sizes_kvasir.csv`):

| Width | Height | Channels | Images |
|---|---|---|---|
| 622 | 530 | 3 | 78 |
| 626 | 547 | 3 | 66 |
| 626 | 546 | 3 | 61 |
| 622 | 529 | 3 | 56 |
| 622 | 528 | 3 | 53 |
| 622 | 531 | 3 | 50 |
| 570 | 530 | 3 | 33 |
| 570 | 531 | 3 | 19 |
| 720 | 576 | 3 | 17 |
| 1348 | 1070 | 3 | 17 |

... and 323 more distinct sizes covering 550 images.

Most frequent sizes, CVC-ClinicDB (full table: `image_sizes_cvc.csv`):

| Width | Height | Channels | Images |
|---|---|---|---|
| 384 | 288 | 3 | 612 |

## 5. Mask values

| Dataset | Mask file mode | Masks with only {0,255} | Masks with other values | Max distinct values in one mask | Binarised: 2 classes | Binarised: 1 class |
|---|---|---|---|---|---|---|
| Kvasir-SEG | L: 1000 | 0 | 1000 | 18 | 1000 | 0 |
| CVC-ClinicDB | RGB: 612 | 474 | 138 | 64 | 612 | 0 |

Unique raw values of every mask are stored in the column `mask_unique_values` of the per-image CSV. Multi-channel masks are reduced to one channel with the maximum over channels before thresholding.

## 6. Empty and near-full masks

- **Kvasir-SEG:** 0 empty mask(s); 0 mask(s) covering > 90% of the image.
- **CVC-ClinicDB:** 0 empty mask(s); 0 mask(s) covering > 90% of the image.

## 7. Polyp area fraction and number of polyps

Area fraction = polyp pixels / all pixels of the binarised mask, in percent.

| Dataset | Mean | Median | Min | Q1 | Q3 | Max |
|---|---|---|---|---|---|---|
| Kvasir-SEG | 15.39 | 11.40 | 0.47 | 5.94 | 20.98 | 81.18 |
| CVC-ClinicDB | 9.17 | 6.83 | 0.34 | 3.63 | 12.73 | 48.96 |

| Dataset | Mean polyps | Median | Max | Masks with 1 | with 2 | with 3 | with 4+ |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 1.21 | 1 | 10 | 819 | 164 | 14 | 3 |
| CVC-ClinicDB | 1.17 | 1 | 11 | 542 | 57 | 8 | 5 |

Polyps are counted as 4-connected components (`scipy.ndimage.label`) of the binarised mask. Small isolated specks left after thresholding noisy masks are counted too, so values above 1 can include annotation noise rather than separate polyps.

## 8. Brightness (for domain-shift analysis)

| Dataset | Grey mean | Grey std | Grey min | Grey max | R mean | G mean | B mean |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG | 97.48 | 13.45 | 48.13 | 151.25 | 142.04 | 82.03 | 60.13 |
| CVC-ClinicDB | 76.19 | 13.86 | 35.70 | 119.96 | 102.21 | 68.70 | 46.95 |

On average Kvasir-SEG images are 21.3 grey levels brighter than CVC-ClinicDB images (97.5 vs 76.2 on a 0-255 scale).

## 9. Duplicates (report only; nothing was deleted)

Exact duplicates (identical file bytes): none, within or across the datasets.

Near-duplicates (perceptual hash, Hamming distance <= 4 of 64 bits). The threshold is deliberately strict; the table shows how many pairs exist at each distance.

| Scope | Hamming 0-1 | 2-3 | 4-5 | 6-7 | 8 | Near-duplicates (<= 4) |
|---|---|---|---|---|---|---|
| across datasets | 0 | 0 | 0 | 1 | 3 | 0 |
| within cvc | 14 | 43 | 126 | 313 | 685 | 183 |
| within kvasir | 3 | 1 | 3 | 22 | 149 | 7 |

**Across the two datasets:** 0 near-duplicate pair(s).

Closest cross-dataset pairs (all listed pairs are within 8 bits; full list in `duplicate_pairs.csv`):

| Image A | Image B | Hamming |
|---|---|---|
| kvasir/cju7d6ux323ze0987xos3srkx.jpg | cvc/576.png | 6 |
| kvasir/cju7flevb2wii08188otgs9p2.jpg | cvc/519.png | 8 |
| kvasir/ck2bxlujamu330725szlc2jdu.jpg | cvc/362.png | 8 |
| kvasir/ck2da7fwcjfis07218r1rvm95.jpg | cvc/193.png | 8 |

**Within CVC-ClinicDB:** 183 near-duplicate pair(s), 181 within the same video sequence.

| Image A | Image B | Hamming | Byte-identical |
|---|---|---|---|
| 2.png | 4.png | 0 | False |
| 48.png | 50.png | 0 | False |
| 62.png | 63.png | 0 | False |
| 112.png | 113.png | 0 | False |
| 194.png | 195.png | 0 | False |
| 194.png | 196.png | 0 | False |
| 195.png | 196.png | 0 | False |
| 198.png | 199.png | 0 | False |
| 200.png | 202.png | 0 | False |
| 220.png | 221.png | 0 | False |

... 173 more in `duplicate_pairs.csv`.

**Within Kvasir-SEG:** 7 near-duplicate pair(s).

| Image A | Image B | Hamming | Byte-identical |
|---|---|---|---|
| cju8amfdtqi4x09871tygrgqe.jpg | cju886ryxnsl50801r93jai7q.jpg | 0 | False |
| cju30bmab08bi0835mvlr6e0r.jpg | cju77k828z46w0871r0avuoo9.jpg | 0 | False |
| cju76lsehyia10987u54vn8rb.jpg | cju772304yw5t0818vbw8kkjf.jpg | 0 | False |
| cju8c5223s8j80850b4kealt4.jpg | cju8dpa89u6l80818dj6lldh9.jpg | 2 | False |
| cju7cue9b232j0801qdzk1ykj.jpg | cju7fob3x301u0755x985pmmq.jpg | 4 | False |
| cju8c1a0ws7o208181c6lbsom.jpg | cju13cgqmnhwn0988yrainhcp.jpg | 4 | False |
| cju8c1a0ws7o208181c6lbsom.jpg | cju83kxitjv340987z09m0ezy.jpg | 4 | False |

- Leakage check, Kvasir-SEG: 2 of 7 within-dataset near-duplicate pairs lie in different splits (e.g. cju30bmab08bi0835mvlr6e0r.jpg (train) ~ cju77k828z46w0871r0avuoo9.jpg (test); cju8c5223s8j80850b4kealt4.jpg (val) ~ cju8dpa89u6l80818dj6lldh9.jpg (test)).
- Leakage check, CVC-ClinicDB: 0 of 183 within-dataset near-duplicate pairs lie in different splits.

## 10. Fixed splits

| Dataset | Method | Seed | Train/val/test pairs | Sequences | Approximation |
|---|---|---|---|---|---|
| Kvasir-SEG | random | 42 | 800/100/100 | n/a | no |
| CVC-ClinicDB | sequence_grouped | 42 | 429/92/91 | 29 | no |

- **Kvasir-SEG:** Random permutation of the sorted file stems with numpy default_rng(seed).
- **CVC-ClinicDB:** Whole video sequences are assigned to one split (seeded shuffle, largest first, greedy balancing). Fractions are matched as closely as whole sequences allow. Achieved fractions: {'train': 0.701, 'val': 0.1503, 'test': 0.1487} (target {'train': 0.7, 'val': 0.15, 'test': 0.15}).

Each split is checked by assertion: the parts are disjoint, together cover every usable pair exactly once, and (sequence split) no sequence appears in two parts. Every dataset's complete pair list is also saved as an external-test list (`splits/*_external_test.json`). Files: `outputs/splits/`.

- Sequence information for CVC-ClinicDB (read from 'metadata.csv'): 29 sequences, 6-26 frames each (median 22).

## 11. Second image copy cross-check (CVC-ClinicDB)

Masks: 612 of 612 compared mask pairs are pixel-identical between the main copy and the second copy. Images: the second copy of 612 of 612 images could not be decoded by Pillow, so images were NOT compared.

## 12. Flagged files

Flag meanings: `corrupt_*` cannot be fully decoded; `size_mismatch` image and mask sizes differ; `empty_mask` no polyp pixel after binarising; `near_full_mask` more than 90% polyp; `not_two_class` binarised mask does not have two classes; `unmatched_*` no partner file; `exact_duplicate`; `near_duplicate_across_datasets`. Within-dataset near-duplicates and noisy-but-valid masks are not listed per file (see sections 5 and 9, `duplicate_pairs.csv`).

No file was flagged.

