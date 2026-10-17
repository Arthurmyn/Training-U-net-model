"""Tests for split logic and the validation assertions."""
from __future__ import annotations

import pytest

from audit import splits
from audit.load import natural_key

STEMS = [str(i) for i in range(1, 101)]
SEQ = {s: str((int(s) - 1) // 7 + 1) for s in STEMS}          # 15 sequences of <= 7 frames, contiguous
FR = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_random_split_sizes_deterministic_and_seeded():
    sizes = {"train": 80, "val": 10, "test": 10}
    a = splits.random_split(STEMS, sizes, 42)
    b = splits.random_split(list(reversed(STEMS)), sizes, 42)       # input order must not matter
    c = splits.random_split(STEMS, sizes, 43)
    assert {k: len(v) for k, v in a.items()} == sizes
    assert a == b and a != c
    splits.validate_split(a, STEMS)


def test_random_split_wrong_total_raises():
    with pytest.raises(ValueError):
        splits.random_split(STEMS, {"train": 80, "val": 10, "test": 5}, 1)


def test_sequence_split_never_shares_a_sequence():
    a = splits.sequence_split(STEMS, SEQ, FR, 42)
    splits.validate_split(a, STEMS, SEQ)
    assert a == splits.sequence_split(STEMS, SEQ, FR, 42)
    owner = {}
    for name, part in a.items():
        for s in part:
            assert owner.setdefault(SEQ[s], name) == name
    sizes = {k: len(v) for k, v in a.items()}
    assert abs(sizes["train"] / 100 - 0.7) < 0.08 and sizes["val"] > 0 and sizes["test"] > 0


def test_block_split_is_contiguous_and_numeric_order():
    a = splits.block_split(STEMS, FR)
    assert [len(a[k]) for k in ("train", "val", "test")] == [70, 15, 15]
    assert a["train"][-1] == "70" and a["val"][0] == "71" and a["test"][-1] == "100"     # '100' after '99', not after '10'
    splits.validate_split(a, STEMS)


def test_validate_rejects_overlap_missing_and_shared_sequence():
    good = splits.random_split(STEMS, {"train": 80, "val": 10, "test": 10}, 1)
    with pytest.raises(AssertionError):
        splits.validate_split({**good, "val": good["val"] + good["test"][:1]}, STEMS)
    with pytest.raises(AssertionError):
        splits.validate_split({**good, "test": good["test"][:-1]}, STEMS)
    bad = {"train": ["1", "2"], "val": ["3"], "test": ["4"]}
    with pytest.raises(AssertionError):
        splits.validate_split(bad, ["1", "2", "3", "4"], {"1": "a", "2": "a", "3": "a", "4": "b"})


def test_write_split_files_is_byte_deterministic(tmp_path):
    from pathlib import Path
    split = splits.random_split(STEMS, {"train": 80, "val": 10, "test": 10}, 5)
    imgs = {s: Path(f"{s}.jpg") for s in STEMS}
    msks = {s: Path(f"{s}.jpg") for s in STEMS}
    for name in ("a", "b"):
        splits.write_split_files(tmp_path / name, "k", "K", split, {"seed": 5}, imgs, msks, None, STEMS)
    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()
    assert '"seed": 5' in (tmp_path / "a" / "k_split.json").read_text()
