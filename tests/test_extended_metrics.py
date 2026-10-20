"""Tests for the extended metrics, the t-based gap statistics, the decision rule and Cohen's kappa."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from train import extended_metrics as em

SIZE = 256


def canvas(*pixels, shape=(32, 32)) -> np.ndarray:
    m = np.zeros(shape, bool)
    for r, c in pixels:
        m[r, c] = True
    return m


def test_counts_dice_iou_precision_recall_known_example():
    gt = canvas(*[(0, c) for c in range(5)])                     # 5 ground-truth pixels
    pred = canvas((0, 0), (0, 1), (0, 2), (4, 4))                # 3 hit, 1 wrong -> TP=3, FP=1, FN=2
    m = em.all_metrics(pred, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 2)
    assert m["dice"] == pytest.approx(0.6667, abs=1e-4)
    assert m["iou"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["miss"] == 0


def test_iou_equals_dice_over_two_minus_dice_on_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.random((16, 16)) > 0.6, rng.random((16, 16)) > 0.6
        m = em.all_metrics(a, b)
        assert m["iou"] == pytest.approx(m["dice"] / (2 - m["dice"]), abs=1e-9)


def test_both_empty_conventions():
    e = np.zeros((SIZE, SIZE), bool)
    m = em.all_metrics(e, e)
    assert m["dice"] == 1.0 and m["iou"] == 1.0 and m["hd95"] == 0.0 and m["miss"] == 1


def test_identical_masks_give_hd95_zero_and_boundary_iou_one():
    gt = np.zeros((SIZE, SIZE), bool)
    gt[60:140, 70:150] = True
    m = em.all_metrics(gt.copy(), gt)
    assert m["hd95"] == 0.0 and m["boundary_iou"] == pytest.approx(1.0) and m["dice"] == 1.0


def test_single_pixels_five_apart_give_hd95_five():
    assert em.hd95(canvas((10, 10)), canvas((10, 15))) == pytest.approx(5.0)
    assert em.hd95(canvas((10, 10)), canvas((15, 10))) == pytest.approx(5.0)


def test_empty_prediction_penalties():
    gt = np.zeros((SIZE, SIZE), bool)
    gt[100:120, 100:120] = True
    pred = np.zeros_like(gt)
    m = em.all_metrics(pred, gt)
    assert m["hd95"] == pytest.approx(256 * math.sqrt(2)) and round(m["hd95"], 2) == 362.04
    assert m["boundary_iou"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0 and m["miss"] == 1 and m["dice"] == 0.0


def test_boundary_uses_four_connectivity_and_counts_image_border():
    full = np.ones((8, 8), bool)
    b = em.boundary(full)
    assert b.sum() == 28 and not b[1:-1, 1:-1].any()             # only the outer ring (border_value=0)
    inner = np.zeros((9, 9), bool)
    inner[2:7, 2:7] = True
    assert em.boundary(inner).sum() == 16                        # 5x5 block: ring of 16
    diag = canvas((3, 3), (4, 4), shape=(9, 9))
    assert em.boundary(diag).sum() == 2                          # diagonal neighbours are not 4-connected: both are boundary


def test_narrow_masks_have_boundary_iou_equal_to_iou():
    gt = np.zeros((64, 64), bool)
    gt[10:13, 10:30] = True                                      # 3 pixels thick (narrower than 2d = 10)
    pred = np.zeros_like(gt)
    pred[11:14, 10:30] = True
    assert (em.boundary_band(gt, 5) == gt).all() and (em.boundary_band(pred, 5) == pred).all()
    assert em.boundary_iou(pred, gt, 5) == pytest.approx(em.iou_from_counts(**em.confusion_counts(pred, gt)))


def test_boundary_band_is_a_ring_for_large_masks():
    m = np.zeros((40, 40), bool)
    m[5:35, 5:35] = True                                         # 30x30 block
    band = em.boundary_band(m, 5)
    assert band.sum() == 30 * 30 - 20 * 20                      # distance <= 5 keeps a ring 5 pixels wide... inner 20x20 removed
    assert not band[15:25, 15:25].any()


def test_boundary_iou_is_less_forgiving_than_iou_for_large_shifted_masks():
    gt = np.zeros((128, 128), bool)
    gt[20:100, 20:100] = True
    pred = np.roll(gt, 6, axis=1)
    assert em.boundary_iou(pred, gt, 5) < em.iou_from_counts(**em.confusion_counts(pred, gt))


# ---------------------------------------------------------------- statistics
import pandas as pd  # noqa: E402

from train import extended_stats as xs  # noqa: E402


def test_t_interval_three_seeds_uses_t_2_df():
    t = xs.t_interval(np.array([0.1, 0.2, 0.3]))
    assert t["t_crit"] == pytest.approx(4.302653, abs=1e-5)
    assert t["mean"] == pytest.approx(0.2) and t["sd"] == pytest.approx(0.1)                  # sample SD, ddof=1
    half = 4.302653 * 0.1 / math.sqrt(3)
    assert (t["low"], t["high"]) == (pytest.approx(0.2 - half, abs=1e-5), pytest.approx(0.2 + half, abs=1e-5))


def test_decision_rule_covers_all_four_outcomes():
    assert xs.decide_delta(0.06, 0.2, 0.05) == "meaningful drop"
    assert xs.decide_delta(-0.2, -0.06, 0.05) == "meaningful increase"
    assert xs.decide_delta(-0.04, 0.04, 0.05) == "equivalence"
    assert xs.decide_delta(-0.02, 0.09, 0.05) == "inconclusive"
    assert xs.decide_delta(0.05, 0.2, 0.05) == "inconclusive"                                  # boundary is not strictly above delta


def test_higher_lower_direction_flags():
    assert xs.HIGHER_IS_BETTER["dice"] and xs.HIGHER_IS_BETTER["boundary_iou"]
    assert not xs.HIGHER_IS_BETTER["hd95"] and not xs.HIGHER_IS_BETTER["miss"]


# ---------------------------------------------------------------- Cohen's kappa
from train import analyze_errors as ae  # noqa: E402


def test_kappa_hand_computed_example():
    # 10 frames. A: five 1s then five 2s. B: 1 1 1 2 2 2 2 2 1 1.
    # agreement on frames 1-3 and 6-8 -> po = 6/10 = 0.6; both raters use 1 and 2 five times each -> pe = 0.5*0.5 + 0.5*0.5 = 0.5
    # kappa = (0.6 - 0.5) / (1 - 0.5) = 0.2
    a = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    b = [1, 1, 1, 2, 2, 2, 2, 2, 1, 1]
    assert ae.cohen_kappa(a, b) == pytest.approx(0.2)


def test_kappa_edge_cases():
    assert ae.cohen_kappa([1, 2, 3, 3], [1, 2, 3, 3]) == pytest.approx(1.0)
    assert ae.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0                                            # pe == 1, full agreement
    assert ae.cohen_kappa([1, 1], [1, 2]) == pytest.approx(0.0)                                   # agreement no better than chance
    assert ae.cohen_kappa([1, 2, 1, 2], [2, 1, 2, 1]) == pytest.approx(-1.0)                      # complete disagreement
    with pytest.raises(ValueError):
        ae.cohen_kappa([1], [1, 2])


def test_parse_category_and_analysis_on_filled_sheets(tmp_path):
    assert ae.parse_category("3") == 3 and ae.parse_category("3.0") == 3 and ae.parse_category("low contrast") == 6
    assert ae.parse_category("") is None and ae.parse_category(None) is None
    with pytest.raises(ValueError):
        ae.parse_category("9")
    stems = [f"{i}" for i in range(1, 11)]
    a = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    b = [1, 1, 1, 2, 2, 2, 2, 2, 1, 1]
    pd.DataFrame({"stem": stems, "category": a, "notes": ""}).to_csv(tmp_path / "annotator_A.csv", index=False)
    pd.DataFrame({"stem": stems[::-1], "category": b[::-1], "notes": ""}).to_csv(tmp_path / "annotator_B.csv", index=False)   # other row order
    assert ae.analyze(tmp_path) == 0
    text = (tmp_path / "summary.md").read_text()
    assert "Cohen's kappa: 0.200" in text and "Disagreements: 4" in text
    cm = pd.read_csv(tmp_path / "confusion_matrix.csv", index_col=0)
    assert cm.loc["A: 1", "B: 1"] == 3 and cm.loc["A: 2", "B: 2"] == 3 and cm.loc["A: 1", "B: 2"] == 2 and cm.loc["A: 2", "B: 1"] == 2
    d = pd.read_csv(tmp_path / "disagreements.csv")
    assert list(d.columns) == ["stem", "category_A", "category_B", "resolved", "notes"] and len(d) == 4


def test_blank_sheets_are_reported_not_crashing(tmp_path):
    pd.DataFrame({"stem": ["1", "2"], "category": ["", ""], "notes": ["", ""]}).to_csv(tmp_path / "annotator_A.csv", index=False)
    pd.DataFrame({"stem": ["1", "2"], "category": ["", ""], "notes": ["", ""]}).to_csv(tmp_path / "annotator_B.csv", index=False)
    assert ae.analyze(tmp_path) == 2 and "not filled" in (tmp_path / "summary.md").read_text()


# ---------------------------------------------------------------- gap tables on synthetic data with a known answer
def _synthetic_extended(seed_offsets=(0.0, 0.1, 0.2)):
    rows = []
    for source, own, ext in (("a", "a_test", "b_external"), ("b", "b_test", "a_external")):
        for k, seed in enumerate((1, 2, 3)):
            for eval_set, base in ((own, 0.8), (ext, 0.5)):
                for i in range(6):
                    v = base + seed_offsets[k] * (1 if eval_set == own else 0)          # own-set Dice grows by seed, external is fixed
                    rows.append({"run": f"{source}_seed{seed}", "source": source, "seed": seed, "eval_set": eval_set,
                                 "kind": "in_domain" if eval_set == own else "external", "target": eval_set.rsplit("_", 1)[0], "dice": v,
                                 "iou": v / (2 - v), "precision": v, "recall": v, "hd95": 10.0 + i, "boundary_iou": v / 2, "miss": 0,
                                 "gt_area": 0.1, "pred_area": 0.1, "stem": f"{eval_set}{i}", "image_file": f"{eval_set}{i}.png",
                                 "sequence_id": None, "target_split": "test"})
    return pd.DataFrame(rows)


def test_gap_tables_known_values_and_preliminary_label(tmp_path):
    from tests.test_train import _project
    from train.config import load_train_config
    cfg = load_train_config(_project(tmp_path)).override(**{"extended.preliminary_sources": ["b"], "analysis.bootstrap_samples": 200})
    df = _synthetic_extended()
    by_seed, summ = xs.gap_tables(df, cfg)
    g = by_seed[(by_seed["direction"] == "A-trained") & (by_seed["metric"] == "dice")].sort_values("seed")
    assert g["gap"].tolist() == pytest.approx([0.3, 0.4, 0.5])                                  # (0.8 + offset) - 0.5 for offsets 0, .1, .2
    row = summ[(summ["direction"] == "A-trained") & (summ["metric"] == "dice")].iloc[0]
    half = 4.302653 * 0.1 / math.sqrt(3)
    assert row["mean_gap"] == pytest.approx(0.4) and row["sd_gap"] == pytest.approx(0.1)
    assert (row["ci_t_low"], row["ci_t_high"]) == (pytest.approx(0.4 - half, abs=1e-5), pytest.approx(0.4 + half, abs=1e-5))
    assert row["better"] == "higher" and summ[(summ["metric"] == "hd95")]["better"].eq("lower").all()
    assert summ[summ["direction"] == "A-trained"]["status"].eq("").all() and summ[summ["direction"] == "B-trained"]["status"].eq("preliminary (E10)").all()
    mbc = xs.metrics_by_condition(df, cfg)
    r = mbc[(mbc["eval_set"] == "a_test") & (mbc["metric"] == "dice")].iloc[0]
    assert r["mean_over_seeds"] == pytest.approx(0.9) and r["sd_over_seeds"] == pytest.approx(0.1) and r["n_seeds"] == 3
    dec = xs.decision_table(summ, [0.05, 0.03, 0.10])
    assert dec[(dec["direction"] == "A-trained") & (dec["delta"] == 0.05)]["decision_t_ci"].iloc[0] == "meaningful drop"
    assert len(dec) == 6 and dec["main_delta"].sum() == 2
