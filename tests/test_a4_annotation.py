"""Annotation documents of the report package (synthetic frames, no data needed)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train import a4_annotation as aa


def _material(tmp_path, n=12):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {"stem": str(i + 1), "sequence_id": str(1 + i % 4), "group": "low_dice" if i < 8 else "reference",
               "polyp_size_bin": aa.BINS[i % 3], "mean_dice": round(float(rng.uniform(0.1, 0.9)), 6)}
        for s in (1, 2):
            row[f"dice_seed{s}"] = row["mean_dice"]
            row[f"gt_area_seed{s}"] = 0.05
            row[f"pred_area_seed{s}"] = 0.0 if i == 0 else 0.04
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "frames.csv", index=False)
    (tmp_path / "frames").mkdir()
    for r in rows:
        (tmp_path / "frames" / f"{r['stem']}.png").write_bytes(b"x")
    for name, off in (("annotator_A.csv", 0), ("annotator_B.csv", 1)):
        pd.DataFrame({"stem": [r["stem"] for r in rows][off:] + [r["stem"] for r in rows][:off], "category": "", "notes": ""}).to_csv(tmp_path / name, index=False)
    (tmp_path / "selection.json").write_text(json.dumps({
        "model": "kvasir-trained", "evaluation_set": "cvc_external", "total_frames": 100, "dice_threshold": 0.5, "frames_below_threshold": 40,
        "sampled_low_dice_frames": 8, "reference_frames": 4, "reference_frames_drawn_from": "x (60 available)", "sampling_seed": 42, "size_bins": "bins"}), encoding="utf-8")
    return pd.DataFrame(rows)


def test_seven_categories_and_checks(tmp_path):
    _material(tmp_path)
    assert len(aa.category_names()) == 7
    c = aa.check_material(tmp_path)
    assert c["same_frames_in_both_sheets"] and c["all_cells_empty"] and c["panels_match_frames"] and c["n_frames"] == 12
    pd.DataFrame({"stem": ["1"], "category": ["3"], "notes": [""]}).to_csv(tmp_path / "annotator_A.csv", index=False)
    assert not aa.check_material(tmp_path)["all_cells_empty"]


def test_selection_counts_come_from_the_files(tmp_path):
    f = _material(tmp_path)
    text = aa.selection_md(tmp_path)
    assert "40 of 100" in text and f"Frames in the annotation set: {len(f)}" in text and "sampling seed 42" in text


def test_failure_cases_rule(tmp_path):
    f = _material(tmp_path)
    cases = aa.pick_cases(f)
    lowest = f.sort_values(["mean_dice", "stem"])["stem"].head(3).tolist()
    assert cases["stem"].head(3).tolist() == lowest
    assert cases["stem"].is_unique and len(cases) >= 4
    assert cases["sequence_id"].tail(len(cases) - 3).is_unique
    md = aa.failure_cases_md(tmp_path)
    assert "- Category: \n" in md and "- Cause: \n" in md and "frames/" in md
    assert "Empty prediction per seed: seed 1: yes" in md      # frame 1 has an empty prediction


def test_guide_lists_seven_categories():
    g = aa.guide_md()
    assert sum(1 for line in g.splitlines() if line[:2] in {f"{i}." for i in range(1, 8)} and "Categories" not in line) >= 7
    assert "Dice values are not available" in g
