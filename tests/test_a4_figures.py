"""Figure list: complete, consistent with the folders and current."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from train import a4_figures as fg

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "all_figures").is_dir(), reason="needs the figure folders")


def test_every_png_is_listed_once_and_file_is_current():
    df = fg.figure_rows(ROOT)
    on_disk = {p.relative_to(ROOT).as_posix() for d in ("final_results/figures", "all_figures") for p in (ROOT / d).rglob("*.png")}
    assert set(df["path"]) == on_disk and df["path"].is_unique
    assert set(df["type"]) <= {"graph", "diagram", "image panels"} and set(df["shows_dataset_frames"]) <= {"yes", "no"}
    committed = pd.read_csv(ROOT / "a4" / "figure_list.csv", dtype=str, keep_default_na=False)
    assert committed.equals(df.astype(str))
    assert (df[df["type"] == "image panels"]["shows_dataset_frames"] == "yes").all()
