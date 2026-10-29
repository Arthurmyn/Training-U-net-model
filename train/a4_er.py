"""Evidence for the expected results ER1-ER8: ``python -m train.a4_er`` writes ``a4/ER_evidence.md`` and ``.csv``.

Observed values come from ``a4/number_ledger.csv``; each criterion is evaluated mechanically (true / false), with no
interpretation. ER6 and ER7 depend on the manual annotation and are reported as "not assessable" while the sheets are empty.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .a4_ledger import ROOT
from .a4_tables import load_ledger

Ledger = Dict[str, tuple]


def _v(led: Ledger, id_: str) -> float:
    return float(led[id_][0])


def annotation_filled(root: Path = ROOT) -> int:
    """Number of non-empty category cells in the two annotation sheets of the final experiment."""
    n = 0
    for name in ("annotator_A.csv", "annotator_B.csv"):
        path = root / "error_analysis_equal" / name
        if path.is_file():
            n += int((pd.read_csv(path, dtype=str, keep_default_na=False)["category"].str.strip() != "").sum())
    return n


def evaluate(led: Ledger, root: Path = ROOT) -> List[dict]:
    rows: List[dict] = []

    def add(er: str, criterion: str, observed: str, verdict: str, ids: str) -> None:
        rows.append({"id": er, "criterion": criterion, "observed": observed, "result": verdict, "ledger_ids": ids})

    def tf(x: bool) -> str:
        return "true" if x else "false"

    e1 = _v(led, "cond.E1.dice.mean_over_seeds")
    add("ER1", "E1 mean Dice in [0.70, 0.85]", f"{e1:.6f}", tf(0.70 <= e1 <= 0.85), "cond.E1.dice.mean_over_seeds")
    gk, gc = _v(led, "gap.kvasir.mean_gap"), _v(led, "gap.cvc.mean_gap")
    add("ER2", "Kvasir-trained gap in [0.03, 0.20]", f"{gk:.6f}", tf(0.03 <= gk <= 0.20), "gap.kvasir.mean_gap")
    add("ER3", "CVC-trained gap in [0.03, 0.25]", f"{gc:.6f}", tf(0.03 <= gc <= 0.25), "gap.cvc.mean_gap")
    for own, ext in (("E1", "E2"), ("E3", "E4")):
        ho, he = _v(led, f"cond.{own}.hd95.mean_over_seeds"), _v(led, f"cond.{ext}.hd95.mean_over_seeds")
        bo, be = _v(led, f"cond.{own}.boundary_iou.mean_over_seeds"), _v(led, f"cond.{ext}.boundary_iou.mean_over_seeds")
        add("ER4", f"HD95 higher on external data, {ext} vs {own}", f"{ext} {he:.6f}; {own} {ho:.6f}", tf(he > ho), f"cond.{ext}.hd95.mean_over_seeds; cond.{own}.hd95.mean_over_seeds")
        add("ER4", f"Boundary IoU lower on external data, {ext} vs {own}", f"{ext} {be:.6f}; {own} {bo:.6f}", tf(be < bo), f"cond.{ext}.boundary_iou.mean_over_seeds; cond.{own}.boundary_iou.mean_over_seeds")
    for cond in ("E1", "E3"):
        sd = _v(led, f"cond.{cond}.dice.sd_over_seeds")
        add("ER5", f"SD of Dice over seeds <= 0.03, {cond}", f"{sd:.6f}", tf(sd <= 0.03), f"cond.{cond}.dice.sd_over_seeds")
    filled = annotation_filled(root)
    state = "not assessable" if filled == 0 else "annotation present"
    for er in ("ER6", "ER7"):
        add(er, "depends on the manual error annotation", f"filled category cells in both sheets: {filled}", state, "")
    for cond, es in (("E2", "cvc_external"), ("E4", "kvasir_external")):
        rho = _v(led, f"spearman.{es}.gt_area_fraction.spearman_rho")
        add("ER8", f"Spearman rho, Dice vs ground-truth polyp area, >= 0.30 on external data, {cond}", f"{rho:.6f}", tf(rho >= 0.30), f"spearman.{es}.gt_area_fraction.spearman_rho")
    return rows


def to_markdown(rows: List[dict]) -> str:
    df = pd.DataFrame(rows)
    lines = ["# Evidence for ER1-ER8", "", "Observed values are read from `a4/number_ledger.csv` (full precision). Each criterion is evaluated mechanically; no interpretation is attached.", "",
             "| id | criterion | observed | result | ledger ids |", "|---|---|---|---|---|"]
    return "\n".join(lines + [f"| {r.id} | {r.criterion} | {r.observed} | {r.result} | {r.ledger_ids} |" for r in df.itertuples()]) + "\n"


def write(root: Path = ROOT) -> List[dict]:
    rows = evaluate(load_ledger(root), root)
    pd.DataFrame(rows).to_csv(root / "a4" / "ER_evidence.csv", index=False)
    (root / "a4" / "ER_evidence.md").write_text(to_markdown(rows), encoding="utf-8")
    return rows


if __name__ == "__main__":
    print(to_markdown(write()))
