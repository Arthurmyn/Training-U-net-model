"""Number ledger: the committed ledger matches its sources and the checker detects changes."""
from __future__ import annotations

import csv
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

from train import a4_ledger as led

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "final_results" / "tables").is_dir(), reason="needs the committed results")


def _checker():
    spec = importlib.util.spec_from_file_location("check_ledger", ROOT / "a4" / "check_ledger.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_ledger_is_current_and_verified():
    committed = pd.read_csv(ROOT / "a4" / "number_ledger.csv", dtype=str, keep_default_na=False)
    fresh = led.build(ROOT).astype(str)
    assert committed.equals(fresh[committed.columns])
    assert committed["id"].is_unique and len(committed) > 300
    assert _checker().check(ROOT) == []


def test_checker_detects_a_changed_value_and_a_missing_source(tmp_path):
    chk = _checker()
    rows = list(csv.DictReader(open(ROOT / "a4" / "number_ledger.csv", newline="", encoding="utf-8")))
    rows[0]["value"] = "999"
    rows[1]["source_file"] = "does/not/exist.json"
    bad = tmp_path / "ledger.csv"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    problems = chk.check(ROOT, bad)
    assert len(problems) == 2 and "ledger 999" in problems[0] and "cannot read" in problems[1]


def test_both_readers_agree_on_every_row():
    chk = _checker()
    for r in led.entries(ROOT):
        assert chk._equal(led.read_cell(ROOT, r["source_file"], r["locator"]), chk.reread(ROOT, r["source_file"], r["locator"])), r["id"]


def test_known_values():
    df = pd.read_csv(ROOT / "a4" / "number_ledger.csv", dtype=str).set_index("id")
    assert float(df.loc["gap.kvasir.mean_gap", "value"]) == pytest.approx(0.166352, abs=1e-6)
    assert df.loc["split.cvc.train", "value"] == "429" and df.loc["audit.cvc.sequences", "value"] == "29"
