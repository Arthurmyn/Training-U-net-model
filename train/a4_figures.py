"""List of every figure of the experiment: ``python -m train.a4_figures`` writes ``a4/figure_list.md`` and ``.csv``.

Covers ``final_results/figures/`` and ``all_figures/``. Descriptions come from the gallery catalogue
(``train.figure_gallery.CATALOGUE``); a figure is marked as showing dataset frames when its catalogue entry says so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

from .figure_gallery import CATALOGUE

ROOT = Path(__file__).resolve().parents[1]
CODE = {c[0]: c for c in CATALOGUE}
# final_results/figures/<file> -> catalogue code of the same figure
FINAL_TO_CODE: Dict[str, str] = {"fig01_study_design.png": "D1", "fig02_dataset_differences.png": "D2", "fig03_dice_by_condition.png": "R1", "fig04_gap_per_seed_dice.png": "R2",
                                 "fig05_all_metrics.png": "R3", "fig06_gap_all_metrics.png": "R4", "fig07_dice_by_polyp_size.png": "F1", "fig08_examples_E2.png": "F6",
                                 "fig08_examples_E4.png": "F7", "fig10_metric_distributions.png": "R5", "figA1_training_curves.png": "T1", "figA2_delta_sensitivity.png": "R6"}
OWN_DESCRIPTION = {"fig09_equal_size_comparison.png": ("Comparison with the earlier preliminary run set", "Dice of the earlier run set (800 vs 429 training images) next to the final experiment", False, "graph")}
FRAMES_MARK = "shows dataset frames"


def _kind(code: str, frames: bool) -> str:
    return "diagram" if code == "D1" else ("image panels" if frames else "graph")


def figure_rows(root: Path = ROOT) -> pd.DataFrame:
    rows: List[dict] = []
    for p in sorted((root / "final_results" / "figures").glob("*.png")):
        if p.name in FINAL_TO_CODE:
            c = CODE[FINAL_TO_CODE[p.name]]
            title, text, frames, kind = c[3], c[4], FRAMES_MARK in c[4], _kind(c[0], FRAMES_MARK in c[4])
        else:
            title, text, frames, kind = OWN_DESCRIPTION[p.name]
        rows.append((p, title, text, kind, frames))
    for p in sorted((root / "all_figures").rglob("*.png")):
        c = CODE[p.name.split("_", 1)[0]]
        frames = FRAMES_MARK in c[4]
        rows.append((p, c[3], c[4], _kind(c[0], frames), frames))
    out = []
    for p, title, text, kind, frames in rows:
        with Image.open(p) as im:
            size = f"{im.width} x {im.height}"
        out.append({"path": p.relative_to(root).as_posix(), "title": title, "description": text.replace(" (shows dataset frames)", ""), "type": kind,
                    "shows_dataset_frames": "yes" if frames else "no", "pixels": size})
    return pd.DataFrame(out)


def to_markdown(df: pd.DataFrame) -> str:
    folder = df["path"].str.startswith("final_results/")
    counts = [f"- Figures in `final_results/figures/`: {int(folder.sum())}", f"- Figures in `all_figures/`: {int((~folder).sum())}", f"- Total: {len(df)}",
              f"- Types: " + ", ".join(f"{k} {v}" for k, v in df["type"].value_counts().sort_index().items()),
              f"- Showing dataset frames: {int((df['shows_dataset_frames'] == 'yes').sum())}"]
    head = ["# Figure list", "", "Every PNG of `final_results/figures/` and `all_figures/`. A figure is marked as showing dataset frames when it contains image content of the datasets (frames with overlays). ", ""] + counts + ["",
            "| path | title | description | type | shows dataset frames | pixels |", "|---|---|---|---|---|---|"]
    return "\n".join(head + [f"| {r.path} | {r.title} | {r.description} | {r.type} | {r.shows_dataset_frames} | {r.pixels} |" for r in df.itertuples()]) + "\n"


def write(root: Path = ROOT) -> pd.DataFrame:
    df = figure_rows(root)
    (root / "a4").mkdir(exist_ok=True)
    df.to_csv(root / "a4" / "figure_list.csv", index=False)
    (root / "a4" / "figure_list.md").write_text(to_markdown(df), encoding="utf-8")
    return df


if __name__ == "__main__":
    print(to_markdown(write()).split("| path")[0])
