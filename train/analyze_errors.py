"""Agreement between the two annotators: ``python -m train.analyze_errors`` (after both sheets are filled).

Writes error_analysis/summary.md, confusion_matrix.csv, category_counts.csv and disagreements.csv (with an empty
``resolved`` column to fill in together).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import load_train_config
from .extended import error_dir

log = logging.getLogger("train")

CATEGORIES: Dict[int, str] = {1: "Missed small polyp", 2: "False detection of folds or reflections", 3: "Incomplete segmentation",
                              4: "Inaccurate boundary", 5: "Complex background", 6: "Low contrast", 7: "Several objects in the frame"}
NAME_TO_ID = {v.lower(): k for k, v in CATEGORIES.items()}


def parse_category(value) -> Optional[int]:
    """Category number 1-7 (also as '3', '3.0' or the category name); None if empty; ValueError if invalid."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.lower() in NAME_TO_ID:
        return NAME_TO_ID[text.lower()]
    try:
        number = int(float(text))
    except ValueError:
        raise ValueError(f"invalid category {value!r}: use 1-7") from None
    if number not in CATEGORIES:
        raise ValueError(f"invalid category {value!r}: use 1-7")
    return number


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa = (po - pe) / (1 - pe) for two raters; own implementation (no external library).

    po is the observed agreement, pe the agreement expected from the two marginal distributions.
    If pe == 1 (both raters always use the same single category) kappa is 1 when they agree, else nan.
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty ratings of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = sorted(set(a) | set(b))
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1.0 - pe)


def confusion_matrix(a: Sequence[int], b: Sequence[int]) -> pd.DataFrame:
    """Counts of (annotator A category, annotator B category) over all 7 categories."""
    labels = list(CATEGORIES)
    m = pd.DataFrame(0, index=labels, columns=labels)
    for x, y in zip(a, b):
        m.loc[x, y] += 1
    m.index = [f"A: {k}" for k in labels]
    m.columns = [f"B: {k}" for k in labels]
    return m


def load_sheet(path: Path) -> pd.Series:
    """Series stem -> category (blank rows dropped)."""
    df = pd.read_csv(path, dtype=str)
    cat = df["category"].map(parse_category)
    return pd.Series(cat.values, index=df["stem"]).dropna().astype(int)


def analyze(directory: Path) -> int:
    """Compute agreement from annotator_A.csv and annotator_B.csv. Returns 0, or 2 if the sheets are not filled."""
    a, b = load_sheet(directory / "annotator_A.csv"), load_sheet(directory / "annotator_B.csv")
    common = sorted(set(a.index) & set(b.index))
    if not common:
        (directory / "summary.md").write_text("# Error analysis summary\n\nThe annotation sheets are not filled in yet (no frame has a category from both annotators).\n", encoding="utf-8")
        log.warning("annotation sheets are empty: nothing to analyse")
        return 2
    ra, rb = [int(a[s]) for s in common], [int(b[s]) for s in common]
    kappa = cohen_kappa(ra, rb)
    agree = sum(x == y for x, y in zip(ra, rb))
    cm = confusion_matrix(ra, rb)
    cm.to_csv(directory / "confusion_matrix.csv")
    counts = pd.DataFrame({"category": [f"{k} {v}" for k, v in CATEGORIES.items()],
                           "annotator_A": [ra.count(k) for k in CATEGORIES], "annotator_B": [rb.count(k) for k in CATEGORIES]})
    dis = pd.DataFrame([{"stem": s, "category_A": int(a[s]), "category_B": int(b[s]), "resolved": "", "notes": ""} for s in common if a[s] != b[s]],
                       columns=["stem", "category_A", "category_B", "resolved", "notes"])
    dis.to_csv(directory / "disagreements.csv", index=False)
    resolved_counts = ""
    if len(dis) and (dis["resolved"].astype(str).str.strip() != "").all():
        final = {s: int(a[s]) for s in common if a[s] == b[s]}
        final.update({r.stem: parse_category(r.resolved) for r in dis.itertuples()})
        counts["final"] = [list(final.values()).count(k) for k in CATEGORIES]
        resolved_counts = "Final counts use the agreed category for every disagreement.\n"
    counts.to_csv(directory / "category_counts.csv", index=False)
    band = ("poor" if kappa < 0 else "slight" if kappa < 0.2 else "fair" if kappa < 0.4 else "moderate" if kappa < 0.6 else "substantial" if kappa < 0.8 else "almost perfect")
    text = ["# Error analysis summary", "",
            f"Frames rated by both annotators: {len(common)} (A: {len(a)}, B: {len(b)}).",
            f"Observed agreement: {agree}/{len(common)} = {agree / len(common):.3f}.",
            f"Cohen's kappa: {kappa:.3f} (Landis and Koch band: {band}).", "",
            "## Category counts", "", counts.to_markdown(index=False) if hasattr(counts, "to_markdown") and _has_tabulate() else counts.to_string(index=False), "",
            resolved_counts, "## Confusion matrix (rows: annotator A, columns: annotator B)", "", cm.to_string(), "",
            f"Disagreements: {len(dis)} (see disagreements.csv, fill the `resolved` column together).", ""]
    (directory / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return 0


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cohen's kappa and summary for the two annotation sheets.")
    ap.add_argument("--config", default="train_config.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    return analyze(error_dir(load_train_config(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())
