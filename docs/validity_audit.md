# Audit of experimental validity

Scope: does the protocol implemented in this repository support the research question in the README (change of per-image Dice when a model trained on one polyp dataset is applied to the other)? Every row names the evidence. "OK" means verified in the code or in committed artifacts on the date of this audit; it does not mean the question is answered.

## Checks

| # | Question | Finding | Evidence | Status |
|---|---|---|---|---|
| 1 | Are train, val and test separated? | Parts are disjoint and cover the dataset exactly once; the external list equals the dataset. Re-checked from the committed files: Kvasir-SEG 800/100/100, CVC-ClinicDB 429/92/91. | `outputs/splits/*.json`; `audit/splits.py` (`validate_split`, asserted at every run) | OK |
| 2 | Patient or video leakage? | CVC-ClinicDB is split by video sequence (29 groups from `metadata.csv`; none crosses a part). There is no patient ID: the original description reports 23 patients for 31 sequences, so patient-level overlap between train and test cannot be excluded. Kvasir-SEG has no grouping information. | `outputs/audit_report.md` section 11; `results/extended` | **Limitation** |
| 3 | Near-duplicate leakage? | Perceptual-hash pairs (Hamming ≤ 4): 7 inside Kvasir-SEG, 183 inside CVC-ClinicDB (181 within one sequence), none between datasets (closest 6 bits). One Kvasir pair was a true duplicate and is forced into one split; one was a hash false positive; five others lie in the same split by chance and were not reviewed by eye. | `outputs/duplicate_pairs.csv`, `config.yaml` (`reviewed_near_duplicates`), `outputs/review_near_duplicates.png` | OK, partly unreviewed |
| 4 | Is validation used only for permitted decisions? | The validation part is used for the best-epoch choice, the learning-rate scheduler and early stopping. Test and external data are not decoded during training. | `train/engine.py` (only `parts["train"]` and `parts["val"]` are loaded; `best.pt` and `scheduler.step` use validation Dice) | OK |
| 5 | Is the external dataset independent of tuning? | Hyper-parameters were chosen once and not tuned on any data. **But** experiment 2 adds `min_epochs: 50`, decided from the training curves of experiment 1 after its test and external results were known. | `train_config.yaml`, `configs/equal_size.yaml`, `docs/research_log.md` | **Limitation** (disclosed amendment) |
| 6 | Is preprocessing consistent across datasets? | One function for both datasets and for training and evaluation: resize to 256 × 256, `value > 127`, pixels / 255; no dataset statistics. The square resize changes the aspect ratio (median w/h 1.16 and 1.33). | `audit/preprocess.py`, `outputs/fig_preprocessing_*.png`, README section 8 | OK, caveat noted |
| 7 | Are the metrics computed correctly? | Unit tests with hand-computed masks (TP = 3, FP = 1, FN = 2 → Dice 0.667; HD95 = 5; empty prediction → 362.04). Recomputing every metric from `best.pt` reproduces the stored Dice (max abs. difference 0.0022, mean ≤ 0.00007) and IoU = Dice/(2 − Dice) to 2e-16. | `tests/test_extended_metrics.py`, `results/extended/acceptance_checks.json` | OK |
| 8 | Are seeds and checkpoints recorded? | Seed, best epoch, configuration, versions and commit are in every `run.json`; best weights are tracked. All runs have `git_dirty: true`, so the exact code of a run cannot be recovered from its commit. | `runs*/*/run.json` | **Limitation** |
| 9 | Can the results be reproduced from artifacts? | The statistics are reproduced exactly: `train.analyze` run into a scratch folder gives byte-identical tables and summary. Training itself was not re-run; GPU runs are not bit-deterministic. | check performed during this audit | OK for analysis; training not verified |
| 10 | Are configurations compared fairly? | **Not in experiment 1**: Kvasir-SEG 800 vs CVC-ClinicDB 429 training images, and the CVC-ClinicDB models did not reach a usable baseline (E10). Experiment 2 equalises the sizes (all 5 seeds finished, independently re-checked on 2026-10-27) but also changes the stopping rule, the number of seeds and, for seeds 2-5, the hardware, so it does not isolate the effect of the training size. | README sections 7 and 12, `docs/errors_log.md` | **Limitation** |
| 10b | Is experiment 2 homogeneous? | Seed 1 was trained on an RTX 3050 under Windows (Python 3.14, PyTorch 2.14), seeds 2-5 on a Tesla T4 under Linux (Python 3.13, PyTorch 2.11). The configuration dictionaries of all 10 runs are identical and every run has 429 training images. Excluding seed 1 changes the Dice gaps by 0.003 (Kvasir-SEG-trained) and 0.001 (CVC-ClinicDB-trained). Recomputed Dice equals the stored Dice (max abs. difference 0.0019). | `runs_equal/*/run.json`, `final_results/tables/acceptance_checks.json` | OK, disclosed |
| 11 | Are the test sets adequate? | The CVC-ClinicDB test set has 91 frames from 5 sequences; its bootstrap interval for Dice is [0.156, 0.575] in experiment 1. Own-test and external sets contain different images, so G mixes domain shift with difficulty differences. | `results/tables/dice_by_condition.csv` | **Limitation** |
| 12 | Are the statistics suitable? | With 3 seeds the t multiplier is 4.303 (df = 2). The cluster bootstrap resamples whole sequences (CVC). Practical thresholds 0.05 / 0.03 / 0.10 come from the project methodology. | `train/extended_stats.py`, `train/analyze.py` | OK, wide intervals |
| 13 | Is the sequence count consistent? | 31 sequences in the original description, 29 in the metadata used here; origin of the difference unresolved. | README section 5, `outputs/audit_report.md` | **Open** |

## Problems found while preparing the README, and what was changed

| Problem | Action |
|---|---|
| Commit hashes in `runs/*/run.json` and in the docs referred to a history that was rewritten (co-author trailer removed). | Added `docs/history_rewrite_map.csv` (old → new hash for 69 commits) and updated the hashes in the docs. `run.json` files are unchanged. |
| A training log contained the Windows user directory. | Replaced by `<home>` in `runs/run_all.log`. The original lines remain in earlier Git history. |
| README numbers could drift from the result files. | The tables are generated by `python -m train.readme_tables` and checked by a test. |
| Example figures show dataset images; CVC-ClinicDB is restricted to research and education, Kvasir-SEG terms were not verified; there was no `LICENSE` file. | The authors chose the MIT license for the code (`LICENSE`). They considered removing the dataset frames, then decided to keep them because the repository is educational and the CVC-ClinicDB terms allow research and educational use; the README states that the copyright remains with the dataset owners and that commercial use is not allowed. The Kvasir-SEG terms are still unverified. |
| `requirements/lock.txt` documents Python 3.9 / PyTorch 2.8, while training ran on Python 3.14 / PyTorch 2.14. | Both environments are listed in the README. Pins were not changed. |
| Documentation is mostly in Russian. | The README is in English and links to the Russian guides. |

## Recommended next experiments

1. Experiment 2 is finished (10 of 10 runs, `final_results/`). Make the CVC-ClinicDB baseline informative before interpreting the reverse direction (next item).
2. Make the CVC-ClinicDB baseline informative: leave-sequence-out or grouped k-fold cross-validation, which also enlarges the test set; stratify by polyp size.
3. Find patient identifiers (or the original sequence list) to split CVC-ClinicDB by patient; resolve the 29 vs 31 sequence discrepancy.
4. Review the five unreviewed Kvasir-SEG near-duplicate pairs.
5. Evaluate at the original resolution as a robustness check of the 256 × 256 metrics.
6. Complete the manual error categorisation (`error_analysis/`) and report Cohen's κ.
7. If transfer is to be studied further: a fine-tuning arm and an intensity-normalisation ablation (a controlled test of brightness as a cause); detection-level metrics if polyp finding is of interest.

## Update: 2026-10-27

All ten equal-size runs are now finished (seeds 1-5 for each source). The table above records the 26 October audit; its one-seed completion statements are historical. Seeds 2-5 were run in Kaggle on Tesla T4/Linux by Nursultan Beisenbek; seed 1 used RTX 3050/Windows. The configuration files in the four imported archives match the repository at import. All runs use 429 training images, but the training environments differ.

Completing the planned runs does not resolve the weak CVC baseline: best validation Dice is below 0.5 for all five seeds, and seeds 2 and 4 select epochs 7 and 16. The CVC preliminary flag is retained. No causal claim is made that more epochs, training size, or one dataset property explains the observed performance. Patient-level separation, sequence-count discrepancy, image permissions and independent manual error annotations remain unresolved.
