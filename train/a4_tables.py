"""Tables of the report package: ``python -m train.a4_tables`` writes ``a4/tables/*.csv`` and ``.md``.

Numeric cells are taken from ``a4/number_ledger.csv`` (so every number has a source); the file inventory is read from the
files themselves. Run ``python -m train.a4_ledger`` first.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .a4_ledger import AUDIT, CONDITIONS, DIRECTIONS, METRICS, ROOT

Ledger = Dict[str, Tuple[str, int]]


def load_ledger(root: Path = ROOT) -> Ledger:
    df = pd.read_csv(root / "a4" / "number_ledger.csv", dtype=str, keep_default_na=False)
    return {r.id: (r.value, int(r.decimals)) for r in df.itertuples()}


def f(led: Ledger, id_: str) -> str:
    """Ledger value formatted with its decimals (text values unchanged)."""
    value, dec = led[id_]
    try:
        return f"{float(value):.{dec}f}"
    except ValueError:
        return value


def raw(led: Ledger, id_: str) -> str:
    return led[id_][0]


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False)])


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def evidence_inventory(root: Path = ROOT) -> pd.DataFrame:
    """Files that carry the evidence, with SHA-256; 'in_manifest' compares with final_results/MANIFEST.json where listed."""
    manifest = json.loads((root / "final_results" / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    groups = [("configuration of the experiment", ["configs/equal_size.yaml", "train_config.yaml", "config.yaml"]),
              ("fixed splits and training subset", sorted(p.relative_to(root).as_posix() for p in (root / "outputs" / "splits").glob("*") if p.is_file())),
              ("dataset audit", ["outputs/audit_report.md", "outputs/summary.json", "outputs/duplicate_pairs.csv", "outputs/per_image_stats_cvc.csv", "outputs/per_image_stats_kvasir.csv"]),
              ("run settings and environment", sorted(p.relative_to(root).as_posix() for p in (root / "runs_equal").glob("*/run.json"))),
              ("run evaluation summaries", sorted(p.relative_to(root).as_posix() for p in (root / "runs_equal").glob("*/eval_summary.json"))),
              ("training logs per epoch", sorted(p.relative_to(root).as_posix() for p in (root / "runs_equal").glob("*/metrics.csv"))),
              ("statistics tables", sorted(p.relative_to(root).as_posix() for p in (root / "final_results" / "tables").glob("*") if p.is_file())),
              ("per-image metrics", sorted(p.relative_to(root).as_posix() for p in (root / "final_results" / "per_image").glob("*.csv"))),
              ("metric definitions", ["final_results/METRICS.md"]),
              ("audit report copy", ["final_results/audit/audit_report.md"]),
              ("file manifest", ["final_results/MANIFEST.json"]),
              ("error-analysis material", ["error_analysis_equal/frames.csv", "error_analysis_equal/selection.json", "error_analysis_equal/SELECTION.md"])]
    rows = []
    for group, files in groups:
        for rel in files:
            p = root / rel
            if not p.is_file():
                continue
            key = rel[len("final_results/"):] if rel.startswith("final_results/") else None
            rows.append({"group": group, "path": rel, "bytes": p.stat().st_size, "sha256": _sha(p),
                         "in_manifest": ("yes" if key in manifest and manifest[key]["sha256"] == _sha(p) else "no") if key in manifest else "n/a"})
    return pd.DataFrame(rows)


def data_preparation(led: Ledger, root: Path = ROOT) -> pd.DataFrame:
    """Operations of the data preparation. Parameters are parsed from the audit report text; counts come from the ledger."""
    text = (root / AUDIT).read_text(encoding="utf-8")
    size = re.search(r"resize to (\d+)x(\d+) \(bilinear for images, nearest neighbour for masks\)", text)
    thr = re.search(r"binarise masks with value > (\d+)", text)
    seed = re.search(r"^Seed (\d+);", text, re.M)
    ham = re.search(r"Hamming distance <= (\d+) of 64 bits", text)
    rows = [
        ("1", "pair images with masks", "matched pairs, Kvasir-SEG / CVC-ClinicDB", f"{f(led, 'audit.kvasir.pairs')} / {f(led, 'audit.cvc.pairs')}", "audit.*.pairs"),
        ("2", "binarise masks", "threshold (value > t)", thr.group(1), AUDIT),
        ("2", "binarise masks", "masks with values other than 0 and 255, Kvasir-SEG / CVC-ClinicDB", f"{f(led, 'audit.kvasir.masks_other_values')} / {f(led, 'audit.cvc.masks_other_values')}", "audit.*.masks_other_values"),
        ("3", "resize", "output size (pixels), images bilinear, masks nearest neighbour", f"{size.group(1)} x {size.group(2)}", AUDIT),
        ("3", "resize", "distinct input sizes, Kvasir-SEG / CVC-ClinicDB", f"{f(led, 'audit.kvasir.distinct_sizes')} / {f(led, 'audit.cvc.distinct_sizes')}", "audit.*.distinct_sizes"),
        ("4", "scale pixel values", "range", "[0, 1]", AUDIT),
        ("5", "near-duplicate search (perceptual hash)", "Hamming distance limit (bits)", ham.group(1), AUDIT),
        ("5", "near-duplicate search (perceptual hash)", "pairs within Kvasir-SEG / within CVC-ClinicDB / across", f"{f(led, 'audit.near_duplicates.kvasir')} / {f(led, 'audit.near_duplicates.cvc')} / {f(led, 'audit.near_duplicates.across')}", "audit.near_duplicates.*"),
        ("6", "fixed split, Kvasir-SEG (random)", "train / val / test images", " / ".join(f(led, f"split.kvasir.{p}") for p in ("train", "val", "test")), "split.kvasir.*"),
        ("6", "fixed split, CVC-ClinicDB (whole video sequences)", "train / val / test frames", " / ".join(f(led, f"split.cvc.{p}") for p in ("train", "val", "test")), "split.cvc.*"),
        ("6", "fixed split, CVC-ClinicDB (whole video sequences)", "video sequences listed in metadata.csv", f(led, "audit.cvc.sequences"), "audit.cvc.sequences"),
        ("6", "fixed splits", "random seed of the audit", seed.group(1), AUDIT),
        ("7", "training subset (Kvasir-SEG)", "images in the training subset (one run, all seeds)", f(led, "run.kvasir_seed1.n_train"), "run.kvasir_seed1.n_train"),
    ]
    return pd.DataFrame(rows, columns=["step", "operation", "parameter", "value", "source"])


def run_table(led: Ledger) -> pd.DataFrame:
    runs = sorted({i.split(".")[1] for i in led if i.startswith("run.")})
    cols = [("seed", "seed"), ("n_train", "train images"), ("n_val", "val images"), ("epochs_run", "epochs"), ("best_epoch", "best epoch"), ("best_val_dice", "best val Dice"), ("stop_reason", "stop reason")]
    env = ["gpu", "platform", "python", "torch", "numpy", "git_dirty"]
    rows = []
    for r in runs:
        row = {"run": r}
        row.update({label: f(led, f"run.{r}.{k}") for k, label in cols})
        row.update({f"own test Dice": f(led, f"run.{r}.own_test.mean_dice"), "own test images": f(led, f"run.{r}.own_test.n_images"),
                    "external Dice": f(led, f"run.{r}.external.mean_dice"), "external images": f(led, f"run.{r}.external.n_images")})
        row.update({k: f(led, f"run.{r}.env.{k}") if f"run.{r}.env.{k}" in led else "" for k in env})
        rows.append(row)
    return pd.DataFrame(rows)


def results_by_condition(led: Ledger) -> pd.DataFrame:
    names = {"E1": "Kvasir-SEG -> Kvasir-SEG test", "E2": "Kvasir-SEG -> CVC-ClinicDB", "E3": "CVC-ClinicDB -> CVC-ClinicDB test", "E4": "CVC-ClinicDB -> Kvasir-SEG"}
    rows = []
    for cond in CONDITIONS:
        row = {"condition": cond, "description": names[cond], "images": f(led, f"cond.{cond}.n_images"), "seeds": f(led, f"cond.{cond}.n_seeds")}
        for m in METRICS:
            row[m] = f"{f(led, f'cond.{cond}.{m}.mean_over_seeds')} ± {f(led, f'cond.{cond}.{m}.sd_over_seeds')}"
        rows.append(row)
    return pd.DataFrame(rows)


def gap_table(led: Ledger) -> pd.DataFrame:
    rows = []
    for d in DIRECTIONS:
        t = "kvasir" if d.startswith("Kvasir") else "cvc"
        row = {"direction": d, "own test Dice": f(led, f"gap.{t}.own_mean"), "external Dice": f(led, f"gap.{t}.external_mean"), "gap": f(led, f"gap.{t}.mean_gap"), "SD of gap": f(led, f"gap.{t}.sd_gap"),
               "t interval": f"[{f(led, f'gap.{t}.ci_t_low')}, {f(led, f'gap.{t}.ci_t_high')}]", "bootstrap interval": f"[{f(led, f'gap.{t}.ci_boot_low')}, {f(led, f'gap.{t}.ci_boot_high')}]"}
        for s in range(1, 6):
            row[f"gap seed {s}"] = f(led, f"gap.{t}.seed{s}")
        for delta in ("0.05", "0.03", "0.10"):
            row[f"delta {delta}, t interval"] = f(led, f"decision.{t}.d{delta}.decision_t_ci")
            row[f"delta {delta}, bootstrap"] = f(led, f"decision.{t}.d{delta}.decision_bootstrap_ci")
        rows.append(row)
    return pd.DataFrame(rows)


def error_indicators(led: Ledger) -> pd.DataFrame:
    rows = []
    for cond, es in CONDITIONS.items():
        row = {"condition": cond, "miss rate": f(led, f"cond.{cond}.miss.mean_over_seeds"), "HD95 (px)": f(led, f"cond.{cond}.hd95.mean_over_seeds")}
        for b in ("small", "medium", "large"):
            row[f"Dice, {b} polyps (n)"] = f"{f(led, f'size.{cond}.{b}.mean')} ± {f(led, f'size.{cond}.{b}.sd')} ({f(led, f'size.{cond}.{b}.n')})"
        sp = f"spearman.{es}.gt_area_fraction.spearman_rho" if f"spearman.{es}.gt_area_fraction.spearman_rho" in led else None
        row["Spearman, Dice vs polyp area"] = f(led, sp) if sp else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    # the Spearman correlation is defined for the external sets only
    df.loc[df["condition"].isin(["E1", "E3"]), "Spearman, Dice vs polyp area"] = "n/a"
    return df


def write_all(root: Path = ROOT) -> List[str]:
    led = load_ledger(root)
    out = root / "a4" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tables = {"evidence_inventory": evidence_inventory(root), "data_preparation": data_preparation(led, root), "run_table": run_table(led),
              "results_by_condition": results_by_condition(led), "gap_table": gap_table(led), "error_indicators": error_indicators(led)}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        extra = ""
        if name == "error_indicators":
            extra = (f"\n\nFrames of the CVC-ClinicDB set with seed-averaged Dice < 0.5 (Kvasir-SEG-trained model): {f(led, 'errors.frames_below_0.5')} of {f(led, 'errors.frames_total')}. "
                     f"Polyp-size bins: small < {f(led, 'size.cut.q1')}, large > {f(led, 'size.cut.q3')} (fraction of the 256 x 256 image).")
        (out / f"{name}.md").write_text(f"# {name.replace('_', ' ').capitalize()}\n\n" + md_table(df) + extra + "\n", encoding="utf-8")
    return list(tables)


if __name__ == "__main__":
    print(write_all())
