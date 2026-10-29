"""Documents for the manual error annotation: ``python -m train.a4_annotation --dir error_analysis_equal``.

Reads ``frames.csv`` and ``selection.json`` written by ``train.error_material`` and writes ``SELECTION.md``,
``annotation_guide.md``, ``failure_cases.md`` and ``CATEGORIES.md``. Every number comes from those files; the annotation
sheets are only read (and checked), never filled in.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .error_material import CATEGORIES_TEXT

ROOT = Path(__file__).resolve().parents[1]
BINS = ("small", "medium", "large")


def category_names() -> List[str]:
    """Category names of the rule table (rows that start with a number)."""
    return [m.group(1).strip() for m in re.finditer(r"^\| \d \| ([^|]+) \|", CATEGORIES_TEXT, re.M)]


def check_material(directory: Path) -> Dict[str, object]:
    """Mechanical checks of the annotation material; every value is a plain fact."""
    a = pd.read_csv(directory / "annotator_A.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(directory / "annotator_B.csv", dtype=str, keep_default_na=False)
    frames = pd.read_csv(directory / "frames.csv", dtype={"stem": str})
    stems = set(frames["stem"])
    panels = {p.stem for p in (directory / "frames").glob("*.png")}
    return {"n_frames": len(stems), "sheet_A_rows": len(a), "sheet_B_rows": len(b),
            "same_frames_in_both_sheets": set(a["stem"]) == set(b["stem"]) == stems,
            "all_cells_empty": bool((a["category"] == "").all() and (b["category"] == "").all() and (a["notes"] == "").all() and (b["notes"] == "").all()),
            "panels_match_frames": panels == stems, "n_categories": len(category_names()),
            "columns_A": list(a.columns), "columns_B": list(b.columns)}


def selection_md(directory: Path) -> str:
    info = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    frames = pd.read_csv(directory / "frames.csv", dtype={"stem": str})
    by_group = frames["group"].value_counts().to_dict()
    cross = pd.crosstab(frames["group"], frames["polyp_size_bin"]).reindex(columns=list(BINS), fill_value=0)
    lines = ["# Selection of the frames for the manual annotation", "",
             f"Model: {info['model']} (seeds 1-5, `runs_equal/`). Evaluation set: `{info['evaluation_set']}` ({info['total_frames']} frames).", "",
             "## Rule", "",
             f"1. Per frame, the Dice values of the five seeds are averaged.",
             f"2. Frames with seed-averaged Dice < {info['dice_threshold']} are candidates: {info['frames_below_threshold']} of {info['total_frames']}.",
             f"3. If there are more candidates than {info['sampled_low_dice_frames']}, a random sample of that size is drawn without replacement (sampling seed {info['sampling_seed']}).",
             f"4. {info['reference_frames']} reference frames are drawn at random (same sampling seed) from the frames with Dice >= {info['dice_threshold']}; {info['reference_frames_drawn_from']}.",
             f"5. Polyp-size bins: {info['size_bins']}.", "",
             "## Counts", "",
             f"- Frames in the annotation set: {len(frames)} (low Dice: {by_group.get('low_dice', 0)}, reference: {by_group.get('reference', 0)}).",
             f"- Video sequences represented: {frames['sequence_id'].nunique()}.", "",
             "| group | " + " | ".join(BINS) + " |", "|---|" + "---|" * len(BINS)]
    lines += [f"| {g} | " + " | ".join(str(int(cross.loc[g, k])) for k in BINS) + " |" for g in cross.index]
    lines += ["", "## Files", "",
              "- `frames/<stem>.png`: panel of one frame (image, ground truth, one prediction per seed). No Dice value is drawn.",
              "- `annotator_A.csv`, `annotator_B.csv`: the same frames in a different shuffled order each; `category` and `notes` are empty.",
              "- `CATEGORIES.md`, `annotation_guide.md`: category rules and procedure.",
              "- `frames.csv`, `selection.json`, `failure_cases.md`: contain Dice values and are not for the annotators.", ""]
    return "\n".join(lines)


def guide_md() -> str:
    names = category_names()
    lines = ["# Annotation guide", "",
             "Two annotators label the same frames independently.", "",
             "1. Open the annotator's own sheet (`annotator_A.csv` or `annotator_B.csv`) and the panels in `frames/`.",
             "2. Look at the panel `frames/<stem>.png` for each row. Dice values are not available to the annotators.",
             "3. Enter one category number (1-7) in the `category` column. One category per frame; the closest one if none fits well. Optional remarks go to `notes`.",
             "4. Do not discuss frames with the other annotator, and do not look at the other sheet, until both sheets are complete.",
             "5. Do not edit the `stem` column or the order of the rows.", "",
             "## Categories", ""]
    lines += [f"{i}. {n}" for i, n in enumerate(names, 1)]
    lines += ["", "The full description of each category is in `CATEGORIES.md`.", ""]
    return "\n".join(lines)


def pick_cases(frames: pd.DataFrame) -> pd.DataFrame:
    """The 3 lowest seed-averaged Dice, then one frame per size bin (lower median of the Dice in the bin) from sequences not yet used."""
    f = frames.copy()
    f["stem"] = f["stem"].astype(str)
    f = f.sort_values(["mean_dice", "stem"]).reset_index(drop=True)
    picked = f.head(3).assign(reason="lowest Dice")
    used_stems, used_seq = set(picked["stem"]), set(picked["sequence_id"].astype(str))
    rows = [picked]
    for name in BINS:
        pool = f[(f["polyp_size_bin"] == name) & ~f["stem"].isin(used_stems) & ~f["sequence_id"].astype(str).isin(used_seq)]
        if pool.empty:
            continue
        row = pool.iloc[[(len(pool) - 1) // 2]].assign(reason=f"median Dice of the {name} bin")
        used_stems.add(row["stem"].iloc[0])
        used_seq.add(str(row["sequence_id"].iloc[0]))
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def failure_cases_md(directory: Path) -> str:
    frames = pd.read_csv(directory / "frames.csv", dtype={"stem": str, "sequence_id": str})
    cases = pick_cases(frames)
    seeds = sorted(int(c.replace("dice_seed", "")) for c in frames.columns if c.startswith("dice_seed"))
    lines = ["# Failure cases", "",
             "Selection rule: the 3 frames with the lowest seed-averaged Dice among the frames of the annotation set (`frames.csv`; ties are broken by the frame name in text order), then one frame for each polyp-size bin "
             "(the lower median by Dice of the bin, taken from a video sequence not used yet). Areas are fractions of the 256 x 256 image. "
             "Category and cause are left blank for the annotators.", ""]
    for _, r in cases.iterrows():
        stem = r["stem"]
        dice = ", ".join(f"seed {s}: {r[f'dice_seed{s}']:.3f}" for s in seeds)
        pred = ", ".join(f"seed {s}: {r[f'pred_area_seed{s}']:.4f}" for s in seeds)
        empty = ", ".join(f"seed {s}: {'yes' if r[f'pred_area_seed{s}'] == 0 else 'no'}" for s in seeds)
        lines += [f"## Frame {stem}", "",
                  f"- Selected as: {r['reason']}",
                  f"- Sequence: {r['sequence_id']}; polyp-size bin: {r['polyp_size_bin']}; group: {r['group']}",
                  f"- Mean Dice over seeds: {r['mean_dice']:.3f}",
                  f"- Dice per seed: {dice}",
                  f"- Ground-truth area: {r['gt_area_seed1']:.4f}",
                  f"- Predicted area per seed: {pred}",
                  f"- Empty prediction per seed: {empty}",
                  f"- Panel: [`frames/{stem}.png`](frames/{stem}.png)",
                  "- Category: ", "- Cause: ", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write SELECTION.md, annotation_guide.md, failure_cases.md and CATEGORIES.md.")
    ap.add_argument("--dir", default="error_analysis_equal")
    args = ap.parse_args(argv)
    d = (ROOT / args.dir).resolve()
    checks = check_material(d)
    bad = [k for k in ("same_frames_in_both_sheets", "all_cells_empty", "panels_match_frames") if not checks[k]]
    if bad or checks["n_categories"] != 7:
        print("annotation material check failed:", bad, checks["n_categories"])
        return 1
    (d / "SELECTION.md").write_text(selection_md(d), encoding="utf-8")
    (d / "annotation_guide.md").write_text(guide_md(), encoding="utf-8")
    (d / "failure_cases.md").write_text(failure_cases_md(d), encoding="utf-8")
    (d / "CATEGORIES.md").write_text(CATEGORIES_TEXT, encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
