# Training-U-net-model
In this experiment we'll train U-net based models on 2 datasets: Kvasir-SEG and CVC-ClinicDB
(train on one, test on the other, and the reverse; Dice per image).

This part of the project is the **dataset audit and fixed splits**. No model is trained here.

## Layout
```
audit.ipynb          main notebook: runs the audit step by step, shows tables and figures
config.yaml          dataset paths (relative to this file), expected counts, seed, thresholds
audit/               package: config, load, checks, splits, report, run (CLI)
tests/               pytest tests on small synthetic images (corrupt file, JPEG-noisy mask, empty mask, unmatched file, ...)
outputs/             audit_report.md, per_image_stats_*.csv, summary.json, splits/, figures
data/                raw data (git-ignored, read-only; never modified by the audit)
```

## Data
Extract the two archives so that `config.yaml` finds them:
```
data/kvasir_seg/Kvasir-SEG/Kvasir-SEG/{images,masks}      (1000 .jpg each)
data/cvc_clinicdb/PNG/{Original,"Ground Truth"}            (612 .png each) + metadata.csv (sequence IDs)
data/cvc_clinicdb/TIF/...                                  (optional second copy, used for a cross-check)
```
Images and masks are matched by filename stem. Change folders in `config.yaml` if your layout differs.

## Run
```bash
pip install -r requirements.txt
python -m audit.run --config config.yaml     # terminal
jupyter notebook audit.ipynb                 # or open the notebook
pytest                                       # tests
```
Running twice gives byte-identical outputs (no timestamps, fixed seed, sorted file order).

## Splits (committed, in `outputs/splits/`)
- **Kvasir-SEG:** 800/100/100, random with seed 42 (`kvasir_split.json|csv`).
- **CVC-ClinicDB:** whole video sequences go to one split (about 70/15/15 of frames), sequence IDs from `metadata.csv`.
  If sequence IDs are missing, the code falls back to contiguous blocks and the report says it is an approximation.
- `*_external_test.json`: the whole dataset, used when a model trained on the other dataset is tested.
- The distribution's own `train.txt` / `val.txt` for Kvasir-SEG (880/120) are not used.

## Notes on the audit
- Near-duplicates use a 64-bit DCT perceptual hash (own implementation, `imagehash` not needed); threshold: Hamming <= 4 (`config.yaml`).
- Masks are binarised with `value > 127`; polyps are counted as 4-connected components.
- Findings and limitations are written to `outputs/audit_report.md` (section "Problems found").
