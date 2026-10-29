"""ER evidence: criteria are evaluated mechanically from ledger values."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from train import a4_er as er

ROOT = Path(__file__).resolve().parents[1]


def _led(**kw):
    base = {"cond.E1.dice.mean_over_seeds": 0.70, "gap.kvasir.mean_gap": 0.20, "gap.cvc.mean_gap": 0.0299, "cond.E1.hd95.mean_over_seeds": 10.0, "cond.E2.hd95.mean_over_seeds": 10.0,
            "cond.E1.boundary_iou.mean_over_seeds": 0.5, "cond.E2.boundary_iou.mean_over_seeds": 0.4, "cond.E3.hd95.mean_over_seeds": 5.0, "cond.E4.hd95.mean_over_seeds": 6.0,
            "cond.E3.boundary_iou.mean_over_seeds": 0.5, "cond.E4.boundary_iou.mean_over_seeds": 0.6, "cond.E1.dice.sd_over_seeds": 0.03, "cond.E3.dice.sd_over_seeds": 0.0301,
            "spearman.cvc_external.gt_area_fraction.spearman_rho": 0.30, "spearman.kvasir_external.gt_area_fraction.spearman_rho": 0.2999}
    base.update(kw)
    return {k: (str(v), 3) for k, v in base.items()}


def test_boundaries_are_inclusive_and_mechanical(tmp_path):
    res = {(r["id"], r["criterion"]): r["result"] for r in er.evaluate(_led(), tmp_path)}
    get = lambda i, text: next(v for (a, b), v in res.items() if a == i and text in b)  # noqa: E731
    assert get("ER1", "E1") == "true" and get("ER2", "Kvasir") == "true" and get("ER3", "CVC") == "false"
    assert get("ER4", "HD95 higher on external data, E2") == "false"          # equal values: not higher
    assert get("ER4", "Boundary IoU lower on external data, E2") == "true"
    assert get("ER4", "HD95 higher on external data, E4") == "true" and get("ER4", "Boundary IoU lower on external data, E4") == "false"
    assert get("ER5", "E1") == "true" and get("ER5", "E3") == "false"
    assert get("ER8", "E2") == "true" and get("ER8", "E4") == "false"
    assert get("ER6", "annotation") == "not assessable"


def test_annotation_state(tmp_path):
    d = tmp_path / "error_analysis_equal"
    d.mkdir()
    pd.DataFrame({"stem": ["1", "2"], "category": ["", "3"], "notes": ["", ""]}).to_csv(d / "annotator_A.csv", index=False)
    assert er.annotation_filled(tmp_path) == 1
    assert {r["result"] for r in er.evaluate(_led(), tmp_path) if r["id"] == "ER6"} == {"annotation present"}


@pytest.mark.skipif(not (ROOT / "a4" / "ER_evidence.csv").is_file(), reason="needs the committed evidence")
def test_committed_file_is_current():
    from train.a4_tables import load_ledger
    fresh = pd.DataFrame(er.evaluate(load_ledger(ROOT), ROOT)).astype(str)
    assert pd.read_csv(ROOT / "a4" / "ER_evidence.csv", dtype=str, keep_default_na=False).equals(fresh)
