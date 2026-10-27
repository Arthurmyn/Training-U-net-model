"""Smoke test of the figure gallery on a tiny synthetic experiment (CPU)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from train import figure_gallery as gal  # noqa: E402
from train.config import load_train_config  # noqa: E402
from train.engine import train_run  # noqa: E402
from train.evaluate import evaluate_run  # noqa: E402
from train.extended import stage_inference  # noqa: E402

from .test_train import _project  # noqa: E402


def test_catalogue_is_consistent():
    codes = [c[0] for c in gal.CATALOGUE]
    names = [c[2] for c in gal.CATALOGUE]
    assert len(codes) == len(set(codes)) and len(names) == len(set(names))
    assert all(n.startswith(c + "_") for c, n in zip(codes, names))
    assert {c[1] for c in gal.CATALOGUE} == {"01_data", "02_training", "03_results", "04_errors", "05_robustness"}


def test_gallery_builds_every_figure_that_has_its_input(tmp_path):
    cfg = load_train_config(_project(tmp_path)).override(**{"seeds": [1, 2], "training.max_epochs": 2})
    for source in ("a", "b"):
        for seed in (1, 2):
            evaluate_run(cfg, train_run(cfg, source, seed, progress=False), progress=False, examples=False)
    stage_inference(cfg, progress=False)
    out = tmp_path / "gallery"
    built = gal.build(cfg, out, with_examples=True)
    gal.write_index(out, built)
    names = {p.relative_to(out).as_posix() for p in out.rglob("*.png")}
    expected = {f"{c[1]}/{c[2]}" for c in gal.CATALOGUE if c[0] in built}
    assert names == expected and len(names) >= 20
    for code in ("D1", "D2", "D3", "D4", "D5", "T1", "T2", "T3", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "F1", "F2", "F3", "F4", "F6", "F7", "B2"):
        assert code in built, code
    assert "F5" not in built and "B1" not in built                      # no sequence IDs and no original-resolution file in the synthetic project
    assert not (out / "_work").exists()
    index = (out / "README.md").read_text(encoding="utf-8")
    assert "R1_dice_by_condition.png" in index and str(tmp_path) not in index
