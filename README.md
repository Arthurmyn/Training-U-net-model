# Cross-Dataset Generalization of U-Net for Colorectal Polyp Segmentation

> **Status: research code in progress, not peer reviewed.** The experiment is complete (5 seeds). The experiment with equal training-set sizes is **trained for all five seeds** (10 of 10 runs; CVC results remain preliminary, see [Results](#12-results)). No claim in this document goes beyond the evidence listed in it.

## 1. Abstract

This repository studies how a U-Net trained for binary colorectal polyp segmentation behaves when it is applied, without fine-tuning, to a second public colonoscopy dataset. The two datasets are Kvasir-SEG (1,000 images) and CVC-ClinicDB (612 video frames). Quality is measured by the Dice coefficient per image, together with IoU, precision, recall, HD95 and Boundary IoU.

In the first experiment (3 seeds), a model trained on 800 Kvasir-SEG images reached a mean Dice of 0.822 on the Kvasir-SEG test set and 0.655 on the whole CVC-ClinicDB set; the generalization gap is 0.167 (95% t-interval over seeds [0.079, 0.255]). In the second experiment, both datasets were trained on the same number of images (429) with 5 seeds: the Kvasir-SEG-trained model reached 0.757 on its own test set and 0.590 on CVC-ClinicDB, a gap of 0.166 (95% t-interval over seeds [0.143, 0.189]), so equalising the training size lowered both scores but left the gap unchanged. The reverse direction cannot be interpreted: the CVC-ClinicDB-trained models reached only 0.348 on their own test set and 0.413 on Kvasir-SEG (gap −0.065), and their validation Dice stayed below 0.5 for every seed (error E10), so no reliable within-dataset baseline exists.

## 2. Research question

**How much does the per-image Dice of a U-Net change when a model trained on one polyp dataset is evaluated on the other dataset, compared with its performance on held-out data of its own training dataset, and does this change exceed a practically relevant threshold?**

The practical thresholds on the Dice gap are δ = 0.05 (primary), and δ = 0.03 and 0.10 (sensitivity analyses). The thresholds are taken from the project methodology and were fixed before the statistics in this repository were computed. The experiment measures the combined effect of everything that differs between the two datasets; it cannot attribute a change to a single factor such as brightness, resolution or annotation style.

## 3. Motivation and background

Segmentation models for colonoscopy are usually reported on a test split of the dataset they were trained on. Frames from the same patient or the same video sequence are nearly identical, so a within-dataset score can be optimistic, and a model may not transfer to other endoscopes, hospitals or annotation protocols. U-Net [1] is a standard baseline architecture for biomedical segmentation; this project uses it as a controlled baseline to quantify the transfer gap between two widely used polyp datasets [2, 3].

## 4. Experimental objectives

1. Build an auditable, reproducible data pipeline: dataset audit, fixed splits, duplicate and leakage checks (`audit/`, `outputs/`).
2. Measure within-dataset and cross-dataset Dice per image for both transfer directions (`train/`, `runs/`).
3. Quantify uncertainty separately between training runs (seeds) and between images (video sequences resampled together).
4. Check whether the comparison is fair when the training sets differ in size (800 vs 429 images) by repeating training with equal sizes (`configs/equal_size.yaml`, in progress).

## 5. Datasets

| | Kvasir-SEG | CVC-ClinicDB |
|---|---|---|
| Content | 1,000 polyp images with binary masks | 612 colonoscopy frames with binary masks |
| Image size | 333 distinct sizes, 332-1920 px wide and 352-1072 px high | 384 × 288 (single size) |
| Mask files | JPEG (all 1,000 masks contain values other than 0/255) | PNG (138 of 612 masks contain values other than 0/255) |
| Grouping | none available | frames from video sequences; the metadata file used here lists **29** sequences (the original description reports 31 sequences from 23 patients [4]; the difference is unresolved) |
| Official source | https://datasets.simula.no/kvasir-seg/ [2] | https://polyp.grand-challenge.org/CVCClinicDB/ [3] |

Audit results (`outputs/audit_report.md`, `outputs/summary.json`): all 1,000 and 612 image-mask pairs were found, no file is corrupt, no mask is empty, and no image-mask size mismatch exists. After binarisation (`value > 127`) every mask has exactly two classes.

**Data terms.** CVC-ClinicDB is restricted to research and educational use and forbidden for commercial use; the images belong to Hospital Clínic, Barcelona, and the masks to the Computer Vision Center, Barcelona (official page, accessed during preparation of this README). The Kvasir-SEG terms could not be verified automatically; check the official page before redistributing anything derived from it. Raw data is **not** in this repository (`data/` is git-ignored). The repository does contain figures that show individual frames of the datasets: qualitative examples (`runs*/*/examples_*.png`, `fig08_examples_*.png`), the annotation panels (`error_analysis/frames/`) and the preprocessing and near-duplicate figures in `outputs/`. They are included for research and educational purposes only. The copyright of the images remains with the dataset owners (Hospital Clínic, Barcelona and the Computer Vision Center for CVC-ClinicDB; the authors of Kvasir-SEG); the figures must not be used commercially. If the Kvasir-SEG terms turn out to be stricter, they should be removed.

**Provenance of the archives used here.** The archives are listed with SHA-256 checksums in `data/data_checksums.sha256`. The CVC-ClinicDB archive contains `metadata.csv` (sequence IDs) and `class_dict.csv` next to the image folders. Their origin is not documented in the archive; they are treated as third-party metadata.

## 6. Methodology and model architecture

- **Model:** U-Net with four down/up-sampling stages, 3×3 convolutions with batch normalisation, transposed-convolution up-sampling and a 1×1 output layer (`train/model.py`); 32 base channels, 7,763,041 trainable parameters; one logit per pixel.
- **Task:** binary semantic segmentation (polyp / background) at 256 × 256 pixels. Metrics describe pixel overlap and boundary agreement; no object-level detection metric (such as per-polyp hit rate) is computed.
- **Loss:** binary cross-entropy plus soft Dice (`train/metrics.py`).
- **Optimiser:** AdamW [5] with a plateau learning-rate scheduler driven by validation Dice.

## 7. Experimental design

| Direction | Training data | Model selection (validation) | Test |
|---|---|---|---|
| **A. Within dataset** | train split of dataset X | val split of dataset X | test split of dataset X |
| **B. Cross dataset (no fine-tuning)** | train split of dataset X | val split of dataset X | **all** images of dataset Y |
| **C. Reverse transfer** | train split of Y | val split of Y | all images of X |

Within-dataset evaluation (A) answers how well the model works on unseen images from the same source. Cross-dataset evaluation (B, C) applies the frozen model to a different source; it estimates robustness to a change of data source. The generalization gap is **G = Dice(own test) − Dice(external)**. The external dataset is never read during training, checkpoint selection or learning-rate scheduling (`train/engine.py` loads only the train and val parts).

**Conditions.** E1 = Kvasir-SEG → Kvasir-SEG test; E2 = Kvasir-SEG → CVC-ClinicDB (whole set); E3 = CVC-ClinicDB → CVC-ClinicDB test; E4 = CVC-ClinicDB → Kvasir-SEG (whole set).

**Splits** (`outputs/splits/`, fixed seed 42, generated by `python -m audit.run`):

| Dataset | Method | Train / val / test | Notes |
|---|---|---|---|
| Kvasir-SEG | random, seeded | 800 / 100 / 100 images | the dataset's own `train.txt` / `val.txt` (880 / 120) are not used |
| CVC-ClinicDB | whole video sequences assigned to one part | 429 / 92 / 91 frames (19 / 5 / 5 sequences) | no sequence appears in two parts (checked) |

**Experiments**

| | Experiment 1 (complete) | Experiment 2, equal size (5 seeds trained) |
|---|---|---|
| Training images | Kvasir-SEG 800, CVC-ClinicDB 429 | 429 and 429 (fixed random Kvasir-SEG subset, `outputs/splits/kvasir_train_subset_429_seed42.json`) |
| Seeds | 1, 2, 3 | 1-5 (all completed) |
| Early stopping | patience 15 | patience 15, not before epoch 50 (`min_epochs`, an amendment decided after seeing experiment 1) |
| Config / output | `train_config.yaml` → `runs/`, `results/` | `configs/equal_size.yaml` → `runs_equal/` |

Results of the two experiments must not be pooled: the training sizes and the stopping rule differ.

## 8. Data preprocessing and augmentation

Preprocessing (`audit/preprocess.py`, identical for both datasets and for training and evaluation): resize to 256 × 256 (Pillow bilinear for images, nearest neighbour for masks), binarise masks with `value > 127`, scale pixels to [0, 1]. No dataset-specific normalisation is used. Resizing to a square changes the aspect ratio (median width/height 1.16 for Kvasir-SEG, 1.33 for CVC-ClinicDB). The preprocessing check (`outputs/fig_preprocessing_*.png`) found a maximum polyp-area change of 0.18 percentage points and no mask that became empty.

Augmentation (training only, applied on the fly, no extra images are stored): horizontal and vertical flip (p = 0.5 each), and with p = 0.5 a random rotation up to ±15° with zoom 0.9-1.1. There is no brightness or colour augmentation.

## 9. Training configuration

| Setting | Value | Setting | Value |
|---|---|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | Batch size | 16 |
| LR schedule | halve after 5 epochs without val-Dice gain | Max epochs | 100 |
| Early stopping | patience 15 on val Dice | Threshold | 0.5 |
| Mixed precision | on (CUDA only) | Selection rule | epoch with highest val Dice |

Hyper-parameters were set by the implementer a priori and were **not tuned**; they are not taken from a published protocol. Every run records its seed, best epoch, configuration, Python and library versions, and the Git commit in `runs*/*/run.json`.

## 10. Evaluation metrics

Computed per image on the 256 × 256 prediction (threshold 0.5) and then averaged (definitions: `results/extended/METRICS.md`): Dice = 2TP/(2TP+FP+FN) (both masks empty → 1), IoU, precision, recall, HD95 between mask boundaries in pixels (362.04 for an empty prediction), Boundary IoU with a 5 px band [6], and the miss rate (share of empty predictions).

**Statistics.** For each condition the mean is first taken over images (per seed). The gap G is computed per seed; the primary 95% interval is t-based over seeds (df = number of seeds − 1), the secondary one is a cluster bootstrap (10,000 resamples) in which whole video sequences are resampled for CVC-ClinicDB. Decision rule on a CI [L, U] and threshold δ: L > δ → meaningful drop; U < −δ → meaningful increase; [L, U] ⊂ [−δ, δ] → equivalence; otherwise inconclusive.

## 11. Training and evaluation commands

```bash
# environment (Python >= 3.9; PyTorch only for training)
pip install -r requirements.txt                  # audit and tests
pip install -r requirements/train.txt            # adds PyTorch; choose the CUDA build for your GPU first, see docs/SETUP_ru.md

# data: place the two archives under data/ as described in "Reproducibility", then verify
python -m audit.run --config config.yaml         # audit, fixed splits, outputs/
python -m train.verify_data                      # file hashes equal the audited data
python -m pytest -q                              # unit tests (CPU only)

# training and evaluation (resumable; results in runs/ or runs_equal/)
python -m train.check_gpu                        # shows whether a GPU is used
python -m train.run_all --smoke                  # 2-epoch check, runs_smoke/
python -m train.run_all                          # experiment 1 (train_config.yaml)
python -m train.run_all --config configs/equal_size.yaml --device cuda   # experiment 2
python -m train.run_all --config configs/equal_size.yaml --device cpu    # CPU alternative (much slower)
python -m train.evaluate --run runs/kvasir_seed1 # evaluate one finished run

# statistics, extended metrics, figures
python -m train.analyze                          # Dice by condition and drop at transfer -> results/
python -m train.extended inference               # recompute all metrics from best.pt (aborts if Dice differs from the original)
python -m train.extended stats && python -m train.extended figures && python -m train.extended report
python -m train.collect --config configs/equal_size.yaml   # everything of experiment 2 in final_results/
python -m train.readme_tables                    # checks the tables in this README against results/
```

Further instructions (in Russian): [`docs/SETUP_ru.md`](docs/SETUP_ru.md) (installation, GPU), [`docs/RUNBOOK_ru.md`](docs/RUNBOOK_ru.md), [`docs/EQUAL_SIZE_RUNBOOK_ru.md`](docs/EQUAL_SIZE_RUNBOOK_ru.md), [`docs/RUNPOD_ru.md`](docs/RUNPOD_ru.md).

## 12. Results

All numbers below are generated from the committed result files (`train/readme_tables.py`) and checked by a test. CVC-ClinicDB-trained rows are marked E10: those models did not reach a usable within-dataset baseline.

### Experiment 1 (3 seeds; Kvasir-SEG 800 vs CVC-ClinicDB 429 training images)

<!-- BEGIN:exp1_results -->
| ID | Trained on (images) | Evaluated on (images) | Dice | IoU | Precision | Recall | HD95 [px] | Boundary IoU | Miss rate |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Kvasir-SEG (800) | Kvasir-SEG test (100) | 0.822 ± 0.003 | 0.739 ± 0.004 | 0.849 ± 0.010 | 0.845 ± 0.013 | 41.6 ± 3.4 | 0.374 ± 0.015 | 0.007 ± 0.006 |
| E2 | Kvasir-SEG (800) | CVC-ClinicDB whole set (612) | 0.655 ± 0.036 | 0.562 ± 0.031 | 0.735 ± 0.016 | 0.698 ± 0.054 | 74.5 ± 6.2 | 0.272 ± 0.021 | 0.063 ± 0.036 |
| E3 | CVC-ClinicDB (429) (under-trained, E10) | CVC-ClinicDB test (91) | 0.378 ± 0.036 | 0.288 ± 0.031 | 0.463 ± 0.065 | 0.411 ± 0.021 | 136.2 ± 10.1 | 0.078 ± 0.018 | 0.212 ± 0.023 |
| E4 | CVC-ClinicDB (429) (under-trained, E10) | Kvasir-SEG whole set (1000) | 0.397 ± 0.007 | 0.277 ± 0.004 | 0.326 ± 0.011 | 0.780 ± 0.054 | 113.2 ± 3.7 | 0.062 ± 0.007 | 0.001 ± 0.002 |

Mean ± sample SD over 3 seeds; each seed value is the mean over the images of the evaluation set.
<!-- END:exp1_results -->

Generalization gap (Dice), own test minus external:

<!-- BEGIN:exp1_gap -->
| Model | Own test | External | Gap G | SD of G | 95% CI (t, df = 2) | 95% CI (cluster bootstrap) | δ = 0.05 | δ = 0.03 | δ = 0.10 |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | 0.822 | 0.655 | 0.167 | 0.035 | [0.079, 0.255] | [0.087, 0.251] | meaningful drop | meaningful drop | inconclusive |
| CVC-ClinicDB-trained (E10) | 0.378 | 0.397 | -0.019 | 0.029 | [-0.091, 0.054] | [-0.241, 0.181] | inconclusive | inconclusive | equivalence |
<!-- END:exp1_gap -->

![Dice by condition](results/extended/paper_figures/fig03_dice_by_condition.png)
![Dice gap per seed](results/extended/paper_figures/fig04_gap_per_seed_dice.png)
![All metrics](results/extended/paper_figures/fig05_all_metrics.png)

**Observations (experiment 1).** (i) Kvasir-SEG-trained models score 0.822 on their own test set and 0.655 on CVC-ClinicDB; the gap's t-interval lies entirely above δ = 0.05 and 0.03 and does not decide δ = 0.10. (ii) The CVC-ClinicDB-trained models reach only 0.378 on their own test set, below their score on Kvasir-SEG, so no within-dataset baseline exists for direction C and **no statement about reverse-direction generalization is supported**. (iii) With 3 seeds the t multiplier is 4.303, so intervals are wide. (iv) The two directions are not comparable because the training sizes differ.

### Experiment 2 (equal training size 429 + 429, 5 seeds)

<!-- BEGIN:equal_pilot -->
| Trained on | Seed | Train images | Epochs | Best epoch | Val Dice | Own test Dice (images) | External Dice (images) |
|---|---|---|---|---|---|---|---|
| CVC-ClinicDB | 1 | 429 | 63 | 48 | 0.484 | 0.364 (91) | 0.427 (1000) |
| CVC-ClinicDB | 2 | 429 | 50 | 7 | 0.460 | 0.273 (91) | 0.385 (1000) |
| CVC-ClinicDB | 3 | 429 | 74 | 59 | 0.481 | 0.380 (91) | 0.424 (1000) |
| CVC-ClinicDB | 4 | 429 | 50 | 16 | 0.461 | 0.332 (91) | 0.403 (1000) |
| CVC-ClinicDB | 5 | 429 | 63 | 48 | 0.482 | 0.389 (91) | 0.425 (1000) |
| Kvasir-SEG | 1 | 429 | 100 | 88 | 0.842 | 0.784 (100) | 0.628 (612) |
| Kvasir-SEG | 2 | 429 | 100 | 85 | 0.764 | 0.741 (100) | 0.576 (612) |
| Kvasir-SEG | 3 | 429 | 85 | 70 | 0.709 | 0.703 (100) | 0.506 (612) |
| Kvasir-SEG | 4 | 429 | 87 | 72 | 0.825 | 0.795 (100) | 0.645 (612) |
| Kvasir-SEG | 5 | 429 | 99 | 84 | 0.792 | 0.760 (100) | 0.597 (612) |

**Status: 10 of 10 planned runs finished.** Per-run values; aggregate intervals are computed separately from all available seeds.
<!-- END:equal_pilot -->

All ten runs are complete. Seed 1 was trained on an RTX 3050 under Windows; seeds 2-5 were run by Nursultan Beisenbek in Kaggle on a Tesla T4 under Linux. Training settings and fixed data splits were retained; hardware and software environments differ and are recorded in each run.json. CVC-trained results remain preliminary: validation Dice remains below 0.5 for every seed, and seeds 2 and 4 select epochs 7 and 16 despite running for 50 epochs. Equal training size and a minimum epoch count have not resolved the weak within-dataset baseline. This experiment changes both Kvasir training size and the stopping rule relative to experiment 1; it does not isolate the effect of training size alone.

The five-seed CPU re-evaluation is collected in [`final_results/README.md`](final_results/README.md), with 9,975 image-run records and all seven metrics. Both numerical acceptance checks passed; maximum absolute Dice deviation from the saved GPU evaluations is below 0.002. At delta 0.05, the Kvasir-trained drop is supported by both the seed-based t interval and the cluster bootstrap; the CVC-trained direction is inconclusive and remains preliminary.


### Experiment 2: aggregate results over 5 seeds

<!-- BEGIN:exp2_results -->
| ID | Trained on (images) | Evaluated on (images) | Dice | IoU | Precision | Recall | HD95 [px] | Boundary IoU | Miss rate |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Kvasir-SEG (429) | Kvasir-SEG test (100) | 0.757 ± 0.036 | 0.656 ± 0.040 | 0.780 ± 0.035 | 0.806 ± 0.029 | 54.2 ± 5.8 | 0.289 ± 0.041 | 0.020 ± 0.010 |
| E2 | Kvasir-SEG (429) | CVC-ClinicDB whole set (612) | 0.590 ± 0.054 | 0.498 ± 0.063 | 0.667 ± 0.085 | 0.639 ± 0.028 | 83.2 ± 6.5 | 0.218 ± 0.053 | 0.091 ± 0.026 |
| E3 | CVC-ClinicDB (429) (under-trained, E10) | CVC-ClinicDB test (91) | 0.348 ± 0.047 | 0.261 ± 0.045 | 0.484 ± 0.049 | 0.353 ± 0.050 | 126.6 ± 13.5 | 0.072 ± 0.020 | 0.180 ± 0.052 |
| E4 | CVC-ClinicDB (429) (under-trained, E10) | Kvasir-SEG whole set (1000) | 0.413 ± 0.018 | 0.291 ± 0.016 | 0.352 ± 0.003 | 0.762 ± 0.041 | 109.9 ± 1.7 | 0.065 ± 0.012 | 0.001 ± 0.001 |

Mean ± sample SD over 5 seeds; each seed value is the mean over the images of the evaluation set.
<!-- END:exp2_results -->

Generalization gap (Dice), own test minus external:

<!-- BEGIN:exp2_gap -->
| Model | Own test | External | Gap G | SD of G | 95% CI (t, df = 4) | 95% CI (cluster bootstrap) | δ = 0.05 | δ = 0.03 | δ = 0.10 |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir-SEG-trained | 0.757 | 0.590 | 0.166 | 0.018 | [0.143, 0.189] | [0.084, 0.252] | meaningful drop | meaningful drop | meaningful drop |
| CVC-ClinicDB-trained (E10) | 0.348 | 0.413 | -0.065 | 0.030 | [-0.102, -0.029] | [-0.256, 0.131] | inconclusive | inconclusive | inconclusive |
<!-- END:exp2_gap -->

Gap of every seed, with the machine that trained it (seed 1 on a laptop GPU under Windows, seeds 2-5 on Kaggle under Linux; the machine is therefore confounded with seed 1 only):

<!-- BEGIN:exp2_gap_by_seed -->
| Seed | Trained on | Kvasir-SEG-trained | CVC-ClinicDB-trained |
|---|---|---|---|
| 1 | RTX 3050 6GB | 0.157 | -0.063 |
| 2 | Tesla T4 | 0.165 | -0.112 |
| 3 | Tesla T4 | 0.198 | -0.044 |
| 4 | Tesla T4 | 0.150 | -0.072 |
| 5 | Tesla T4 | 0.162 | -0.036 |
| mean of seeds 2, 3, 4, 5 | same machine for all of them | 0.169 | -0.066 |
<!-- END:exp2_gap_by_seed -->

Kvasir-SEG-trained model in the two experiments (not a clean comparison: experiment 2 changes the training size, the stopping rule, the number of seeds and, for seeds 2-5, the hardware):

<!-- BEGIN:size_comparison -->
|  | Kvasir-SEG training images | Seeds | Own test Dice | CVC-ClinicDB Dice | Gap G | 95% CI (t over seeds) |
|---|---|---|---|---|---|---|
| Experiment 1 | 800 | 3 | 0.822 | 0.655 | 0.167 | [0.079, 0.255] |
| Experiment 2 | 429 | 5 | 0.757 | 0.590 | 0.166 | [0.143, 0.189] |
<!-- END:size_comparison -->

**Observations (experiment 2).** (i) Equalising the training sizes lowered both Kvasir-SEG-trained scores (own test 0.822 → 0.757, external 0.655 → 0.590) but left the gap at 0.166 (experiment 1: 0.167). This is consistent with the gap not being caused by the 800 vs 429 difference in training size alone; it does not isolate that factor, because the stopping rule and the hardware also changed. The gap's t-interval [0.143, 0.189] lies entirely above δ = 0.10. (ii) The CVC-ClinicDB-trained models score 0.348 on their own test set and 0.413 on Kvasir-SEG; they do better on the other dataset than on their own test set. This points to a weak within-dataset baseline (validation Dice 0.460-0.484 for all seeds, selected epochs between 7 and 59), not to good generalization; **no statement about reverse-direction generalization is supported.** (iii) The t-interval over seeds describes variation between training runs for fixed test sets. The cluster bootstrap, which also resamples images and video sequences, is wider ([0.084, 0.252] for the Kvasir-SEG-trained model, [-0.256, 0.131] for the CVC-ClinicDB-trained models) and does not decide the CVC-ClinicDB direction.

## 13. Error analysis

Quantitative indicators from experiment 1 (nothing here is a manual annotation):

<!-- BEGIN:exp1_errors -->
- Kvasir-SEG-trained model on CVC-ClinicDB (E2): 165 of 612 frames have seed-averaged Dice < 0.5.
- Share of images with an empty prediction (miss rate): E1 0.7%, E2 6.3%, E3 21.2%, E4 0.1%.
- Spearman ρ between per-image Dice and ground-truth polyp area, Kvasir-SEG-trained on CVC-ClinicDB: 0.13 (n = 612; descriptive).
- Spearman ρ between per-image Dice and ground-truth polyp area, CVC-ClinicDB-trained on Kvasir-SEG: 0.80 (n = 1000; descriptive).

Dice by ground-truth polyp size (bins are quartiles of the area at 256 px over all evaluated images; mean ± SD over seeds; n = images):

| ID | small (< 4.4% of image) | medium (4.4-17.1%) | large (> 17.1%) |
|---|---|---|---|
| E1 | 0.721 ± 0.057 (n = 13) | 0.846 ± 0.011 (n = 50) | 0.825 ± 0.014 (n = 37) |
| E2 | 0.585 ± 0.017 (n = 200) | 0.689 ± 0.050 (n = 320) | 0.687 ± 0.032 (n = 92) |
| E3 | 0.215 ± 0.040 (n = 34) | 0.446 ± 0.039 (n = 51) | 0.731 ± 0.082 (n = 6) |
| E4 | 0.137 ± 0.012 (n = 168) | 0.349 ± 0.011 (n = 508) | 0.606 ± 0.017 (n = 324) |
<!-- END:exp1_errors -->

![Dice by polyp size](results/extended/paper_figures/fig07_dice_by_polyp_size.png)

![Examples E2](results/extended/paper_figures/fig08_examples_E2.png)

Dice is lowest for small polyps in every condition. The CVC-ClinicDB-trained models are weakest there (0.215 on E3, 0.137 on E4) and return an empty prediction for 21.2% of their own test frames; the Kvasir-SEG-trained models show a smaller size dependence (0.585 on small vs 0.689 on medium polyps in E2). **Manual categorisation of failure cases is pending**: sheets and rules are prepared in `error_analysis/` (150 frames, two blank annotator sheets, `CATEGORIES.md`); after both are filled run `python -m train.analyze_errors` for Cohen's κ.

## 14. Reproducibility

| Item | State |
|---|---|
| Seeds | recorded in `run.json`; Python, NumPy, PyTorch seeded; GPU kernels are not bit-deterministic unless `training.deterministic: true` |
| Checkpoints | best epoch by validation Dice: `runs*/<dataset>_seed<N>/best.pt` (16 files of 31 MB, tracked in Git; the repository pack is about 293 MB, so consider Git LFS or release assets before publishing) |
| Training environment | Experiment 1 and equal-size seed 1: Windows 11, RTX 3050, Python 3.14.3, PyTorch 2.14.1+cu126, NumPy 2.4.3. Equal-size seeds 2-5: Kaggle Linux, Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3. |
| Analysis environment | Original analysis: macOS (Apple Silicon), Python 3.9.6, PyTorch 2.8.0 (`requirements/lock.txt`). Equal-size analysis on 2026-10-27: Linux CPU; exact versions are recorded in `final_results/MANIFEST.json`. |
| Commit of each run | `git_commit` in `run.json`; original runs and equal-size seed 1 have `git_dirty: true`. In Kaggle, `kvasir_seed2` has `git_dirty: false`, and subsequent runs have `true`; that flag alone does not establish which files changed. For `runs/` the recorded hashes refer to the history before it was rewritten; use [`docs/history_rewrite_map.csv`](docs/history_rewrite_map.csv) to find the current hash |
| Re-computation | recomputed Dice reproduces the stored per-image Dice (max abs. difference 0.0022, `results/extended/acceptance_checks.json`); `train.analyze` reproduces `results/tables/` identically from the stored per-image files |
| Hardware | training: GPU recommended (about 10-20 s per epoch on the RTX 3050); `--device cpu` works but is much slower |

**Data placement** (paths in `config.yaml`):

```
data/kvasir_seg/Kvasir-SEG/Kvasir-SEG/{images,masks}      1000 .jpg each
data/cvc_clinicdb/PNG/{Original,"Ground Truth"}            612 .png each, plus metadata.csv
```

## 15. Repository structure

```
config.yaml, train_config.yaml, configs/   settings (audit; training; equal-size experiment)
audit/       dataset audit, fixed splits, duplicate and leakage checks
train/       model, training, evaluation, statistics, extended metrics, figures, collection
tests/       unit and end-to-end tests on small synthetic data (CPU)
notebooks/   audit.ipynb (the audit step by step)
outputs/     audit report, per-image statistics, fixed splits
runs/, runs_equal/   one folder per trained model: logs, per-image Dice, best.pt
results/     statistics, extended metrics, figures;  results/extended/paper_figures/
error_analysis/   frames and annotation sheets
docs/        instructions (Russian), research log, error log, validity audit
```

## 16. Limitations and threats to validity

The full audit with evidence and recommended fixes is in [`docs/validity_audit.md`](docs/validity_audit.md). The main points:

1. **CVC-ClinicDB-trained models are not usable as a baseline** (E10): in both experiments their own-test Dice is below their Dice on the other dataset, and validation Dice stays below 0.5. Experiment 1 compares unequal training sizes; experiment 2 equalises them but also changes the stopping rule and the hardware.
2. **Leakage control is partial.** Kvasir-SEG has no patient or sequence information; CVC-ClinicDB is split by video sequence, not by patient (23 patients for 29-31 sequences), so patient-level overlap between train and test cannot be excluded. Near-duplicates were checked with a perceptual hash (7 pairs inside Kvasir-SEG, 183 inside CVC-ClinicDB, none between the datasets at a distance of at most 4 bits); only two pairs were reviewed by hand.
3. **Small and correlated test sets.** The CVC-ClinicDB test set has 91 frames from 5 sequences; its bootstrap interval is very wide.
4. **Protocol amendment after seeing results.** `min_epochs: 50` in experiment 2 was chosen from the training curves of experiment 1, but after its test and external results were already known; it is documented as an amendment, not as a pre-specified rule.
5. **Own-test and external sets are different images.** G mixes domain shift with differences in difficulty between the evaluation sets.
6. **Metrics are computed at 256 × 256** after an aspect-ratio-changing resize, not at the original resolution.
7. **Experiment 1 has three seeds; experiment 2 has five, trained on two different machines** (seed 1: RTX 3050 under Windows, seeds 2-5: Tesla T4 under Linux, different Python and PyTorch versions). Excluding seed 1 changes the gaps by less than 0.005 (table above), but hardware and seed cannot be separated for seed 1. The t-interval over seeds ignores the finite size of the test sets; the cluster bootstrap is the more conservative statement.
8. The sequence count (29 vs 31) and the origin of `metadata.csv` are unresolved.

## 17. References

1. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, LNCS 9351, pp. 234-241. arXiv:[1505.04597](https://arxiv.org/abs/1505.04597).
2. D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, H. D. Johansen. *Kvasir-SEG: A Segmented Polyp Dataset.* MultiMedia Modeling (MMM) 2020, LNCS 11962, pp. 451-462. DOI [10.1007/978-3-030-37734-2_37](https://doi.org/10.1007/978-3-030-37734-2_37); arXiv:[1911.07069](https://arxiv.org/abs/1911.07069).
3. J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, F. Vilariño. *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians.* Computerized Medical Imaging and Graphics 43 (2015) 99-111. DOI [10.1016/j.compmedimag.2015.02.007](https://doi.org/10.1016/j.compmedimag.2015.02.007). The CVC-ClinicDB page asks to cite this paper.
4. D. Vázquez, J. Bernal, et al. *A Benchmark for Endoluminal Scene Segmentation of Colonoscopy Images.* Journal of Healthcare Engineering 2017, 4037190. DOI [10.1155/2017/4037190](https://doi.org/10.1155/2017/4037190); arXiv:[1612.00799](https://arxiv.org/abs/1612.00799). Source of the figures "31 sequences, 23 patients".
5. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:[1711.05101](https://arxiv.org/abs/1711.05101).
6. B. Cheng, R. Girshick, P. Dollár, A. C. Berg, A. Kirillov. *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation.* CVPR 2021. DOI [10.1109/CVPR46437.2021.01508](https://doi.org/10.1109/CVPR46437.2021.01508); arXiv:[2103.16562](https://arxiv.org/abs/2103.16562).

## 18. Citation

Authors: Artur Jaxygaliyev, Nursultan Beisenbek. If you use this code, cite the repository using [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button), and cite the dataset publications [2, 3] and U-Net [1]. The repository has no DOI and has not been published in a peer-reviewed venue.

## 19. License

The source code is released under the [MIT License](LICENSE). The datasets are **not** covered by it and keep their own terms (see [Datasets](#5-datasets)); the trained weights and figures derived from the datasets should be used under those terms.
