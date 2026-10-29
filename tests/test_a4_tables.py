"""Report tables: committed files are current and every numeric cell comes from the ledger."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from train import a4_tables as tb

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "a4" / "number_ledger.csv").is_file(), reason="needs the ledger")


def test_committed_tables_are_current():
    led = tb.load_ledger(ROOT)
    fresh = {"data_preparation": tb.data_preparation(led, ROOT), "run_table": tb.run_table(led), "results_by_condition": tb.results_by_condition(led),
             "gap_table": tb.gap_table(led), "error_indicators": tb.error_indicators(led), "evidence_inventory": tb.evidence_inventory(ROOT)}
    for name, df in fresh.items():
        committed = pd.read_csv(ROOT / "a4" / "tables" / f"{name}.csv", dtype=str, keep_default_na=False)
        assert committed.equals(df.astype(str)), name


def test_cells_equal_ledger_values():
    led = tb.load_ledger(ROOT)
    res = tb.results_by_condition(led).set_index("condition")
    assert res.loc["E2", "dice"] == f"{float(led['cond.E2.dice.mean_over_seeds'][0]):.3f} ± {float(led['cond.E2.dice.sd_over_seeds'][0]):.3f}"
    gap = tb.gap_table(led).set_index("direction")
    assert gap.loc["Kvasir-SEG-trained", "gap"] == f"{float(led['gap.kvasir.mean_gap'][0]):.3f}"
    runs = tb.run_table(led)
    assert len(runs) == 10 and set(runs["train images"]) == {"429"}


def test_inventory_hashes_match_the_manifest():
    inv = tb.evidence_inventory(ROOT)
    assert (inv["in_manifest"] != "no").all() and (inv["in_manifest"] == "yes").sum() > 5
    assert inv["sha256"].str.len().eq(64).all()
