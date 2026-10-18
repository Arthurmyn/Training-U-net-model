"""End-to-end tests: config loading, run twice -> identical outputs, graceful fallbacks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from audit.config import load_config
from audit.run import run_audit, write_outputs

from .conftest import make_dataset


def _write_config(base: Path, with_sequences: bool) -> Path:
    make_dataset(base / "data" / "a", n_good=20)
    make_dataset(base / "data" / "b", n_good=20)
    if with_sequences:
        rows = ["frame_id,sequence_id"] + [f"g{i:02d},{i // 4}" for i in range(20)]
        (base / "data" / "b" / "meta.csv").write_text("\n".join(rows) + "\n")
    cfg = {
        "seed": 7, "output_dir": "out",
        "thresholds": {"mask_binarize": 127, "full_mask_fraction": 0.9, "near_duplicate_hamming": 4},
        "datasets": {
            "a": {"name": "A", "root": "data/a", "image_dir": "images", "mask_dir": "masks", "expected_count": 27,
                  "split": {"method": "random", "sizes": {"train": 20, "val": 3, "test": 3}}},
            "b": {"name": "B", "root": "data/b", "image_dir": "images", "mask_dir": "masks", "expected_count": 27,
                  "split": {"method": "sequence", "fractions": {"train": 0.6, "val": 0.2, "test": 0.2}}},
        },
    }
    if with_sequences:
        cfg["datasets"]["b"]["sequence_metadata"] = {"file": "meta.csv", "stem_column": "frame_id", "sequence_column": "sequence_id"}
    p = base / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_config_paths_are_relative_to_config_file(tmp_path):
    p = _write_config(tmp_path, True)
    cfg = load_config(p)
    assert cfg.datasets["a"].root == (tmp_path / "data" / "a").resolve()
    assert cfg.output_dir == (tmp_path / "out").resolve()


def test_full_run_twice_gives_identical_outputs(tmp_path):
    p = _write_config(tmp_path, True)
    snapshots = []
    for _ in range(2):
        r = run_audit(load_config(p), progress=False)
        write_outputs(r)
        out = tmp_path / "out"
        snapshots.append({f.relative_to(out).as_posix(): f.read_bytes() for f in sorted(out.rglob("*")) if f.is_file()})
    assert snapshots[0] == snapshots[1]
    names = set(snapshots[0])
    for need in ("audit_report.md", "summary.json", "per_image_stats_a.csv", "per_image_stats_b.csv",
                 "fig_polyp_area_fraction.png", "fig_image_sizes.png", "fig_brightness_hist.png",
                 "splits/a_split.json", "splits/b_split.csv", "splits/a_external_test.json",
                 "fig_preprocessing_a.png", "fig_preprocessing_b.png"):
        assert need in names
    report = snapshots[0]["audit_report.md"].decode()
    assert "corrupt.png" in report and "only_image.png" in report and "only_mask.png" in report
    for name, data in snapshots[0].items():                              # no absolute paths in any text output
        if name.endswith((".md", ".csv", ".json")):
            assert str(tmp_path) not in data.decode(), name
    summ = json.loads(snapshots[0]["summary.json"])
    assert summ["datasets"]["a"]["counts"]["matches_expected"] is False
    assert summ["datasets"]["a"]["preprocessing"]["size"] == 256 and summ["datasets"]["a"]["preprocessing"]["mask_values"] == [0, 1]


def test_splits_exclude_unreadable_and_cover_each_pair_once(tmp_path):
    r = run_audit(load_config(_write_config(tmp_path, True)), progress=False)
    a = r.datasets["a"]
    assert "corrupt" not in a.usable_stems and "corrupt" in a.split_meta["excluded_unreadable_pairs"]
    flat = [s for part in a.split.values() for s in part]
    assert sorted(flat) == sorted(a.usable_stems) and len(flat) == len(set(flat))
    b = r.datasets["b"]
    assert b.split_meta["method"] in ("sequence_grouped", "contiguous_blocks")


def test_missing_sequence_info_falls_back_and_says_so(tmp_path):
    r = run_audit(load_config(_write_config(tmp_path, False)), progress=False)
    b = r.datasets["b"]
    assert b.split_meta["method"] == "contiguous_blocks" and b.split_meta["approximate"]
    write_outputs(r)
    assert "APPROXIMATION" in (tmp_path / "out" / "audit_report.md").read_text()


def test_wrong_split_sizes_are_reported_not_guessed(tmp_path):
    p = _write_config(tmp_path, True)
    raw = yaml.safe_load(p.read_text())
    raw["datasets"]["a"]["split"]["sizes"] = {"train": 800, "val": 100, "test": 100}
    p.write_text(yaml.safe_dump(raw))
    r = run_audit(load_config(p), progress=False)
    assert r.datasets["a"].split is None and r.datasets["a"].split_error
    write_outputs(r)
    assert "split not created" in (tmp_path / "out" / "audit_report.md").read_text()


def test_sequence_integrity_detects_crossing_and_reports_cross_sequence_pairs():
    from audit.run import sequence_integrity
    seqs = {"1": "a", "2": "a", "3": "b", "4": "b"}
    ok = sequence_integrity(seqs, {"train": ["1", "2"], "val": ["3", "4"], "test": []}, ["1", "2", "3", "4"], pd.DataFrame(
        {"a_file": ["2.png"], "b_file": ["3.png"], "hamming": [2]}))
    assert ok["sequences_crossing_splits"] == [] and ok["frame_ranges_contiguous"] is True
    assert ok["cross_sequence_near_duplicates"][0]["split_a"] == "train" and ok["cross_sequence_near_duplicates"][0]["split_b"] == "val"
    bad = sequence_integrity(seqs, {"train": ["1", "3"], "val": ["2", "4"], "test": []}, ["1", "2", "3", "4"],
                             pd.DataFrame({"a_file": [], "b_file": [], "hamming": []}))
    assert bad["sequences_crossing_splits"] == ["a", "b"]
    gap = sequence_integrity({"1": "a", "5": "a"}, {"train": ["1", "5"], "val": [], "test": []}, ["1", "5"],
                             pd.DataFrame({"a_file": [], "b_file": [], "hamming": []}))
    assert gap["frame_ranges_contiguous"] is False


def _clean_sequence_project(base: Path, complete: bool) -> Path:
    make_dataset(base / "data" / "c", n_good=24, defects=False)
    ids = range(24) if complete else range(20)             # incomplete metadata -> block fallback
    rows = ["frame_id,sequence_id"] + [f"g{i:02d},{i // 4}" for i in ids]
    (base / "data" / "c" / "meta.csv").write_text("\n".join(rows) + "\n")
    cfg = {"seed": 3, "output_dir": "out", "datasets": {"c": {
        "name": "C", "root": "data/c", "image_dir": "images", "mask_dir": "masks", "expected_count": 24,
        "sequence_metadata": {"file": "meta.csv", "stem_column": "frame_id", "sequence_column": "sequence_id"},
        "split": {"method": "sequence", "fractions": {"train": 0.5, "val": 0.25, "test": 0.25}}}}}
    p = base / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_complete_sequence_ids_give_no_crossing_sequences(tmp_path):
    c = run_audit(load_config(_clean_sequence_project(tmp_path, True)), progress=False).datasets["c"]
    assert c.split_meta["method"] == "sequence_grouped"
    assert c.sequence_info["integrity"]["sequences_crossing_splits"] == []
    assert c.sequence_info["integrity"]["sequences_checked"] == 6


def test_block_fallback_is_caught_by_the_sequence_check(tmp_path):
    c = run_audit(load_config(_clean_sequence_project(tmp_path, False)), progress=False).datasets["c"]
    assert c.split_meta["method"] == "contiguous_blocks"
    crossing = c.sequence_info["integrity"]["sequences_crossing_splits"]
    assert crossing == ["4"]       # blocks 12/6/6 cut sequence 4 (frames 16-19) at frame 18
