"""Number ledger of the report package: ``python -m train.a4_ledger`` writes ``a4/number_ledger.csv`` and ``.md``.

Every row names a value, the committed file it is read from and a locator that finds the cell again. Nothing is typed by
hand: values are read from ``final_results/``, ``runs_equal/`` and the audit report. ``a4/check_ledger.py`` re-reads every
cell with its own implementation of the locators and fails on any mismatch.

Locators:  ``csv?col=val&col=val#column``  (rows selected by equality, exactly one row must match),
``json#a.b.c``  (dotted path),  ``md?section=<heading prefix>&row=<first cell>#<column>[@i]``  (table cell of a Markdown
report; ``@i`` takes the i-th part of a cell such as ``800/100/100``),  ``derived#name(arg=val,...)``  (computed from the
per-image file; the formulas are the ones of ``train.readme_tables``).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FR = "final_results"
TABLES = f"{FR}/tables"
PER_IMAGE = f"{FR}/per_image/per_image_metrics_extended.csv"
AUDIT = f"{FR}/audit/audit_report.md"
CONDITIONS = {"E1": "kvasir_test", "E2": "cvc_external", "E3": "cvc_test", "E4": "kvasir_external"}
DIRECTIONS = ("Kvasir-SEG-trained", "CVC-ClinicDB-trained")
METRICS = ("dice", "iou", "precision", "recall", "hd95", "boundary_iou", "miss")
COLUMNS = ["id", "value", "meaning", "source_file", "locator", "unit", "decimals", "derived"]


# ---------------------------------------------------------------- reading cells
def _split(locator: str):
    head, _, column = locator.partition("#")
    kind, _, query = head.partition("?")
    return kind, dict(p.split("=", 1) for p in query.split("&") if p), column


def _md_table(text: str, section: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## " + section))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    out: List[Dict[str, str]] = []
    header: Optional[List[str]] = None
    for line in lines[start:end]:
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None:
            header = cells
            continue
        out.append(dict(zip(header, cells)))
    return out


def _same(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a == b


def _derived(root: Path, source: str, name: str, args: Dict[str, str]) -> float:
    df = pd.read_csv(root / source, dtype={"sequence_id": str})
    if name == "frames_below":
        g = df[(df["source"] == "kvasir") & (df["eval_set"] == "cvc_external")].groupby("stem")["dice"].mean()
        return float((g < float(args["thr"])).sum())
    if name == "frames_total":
        return float(df[(df["source"] == "kvasir") & (df["eval_set"] == "cvc_external")]["stem"].nunique())
    area = df.drop_duplicates(["eval_set", "stem"])["gt_area"]
    q1, q3 = float(area.quantile(0.25)), float(area.quantile(0.75))
    if name == "size_cut":
        return q1 if args["which"] == "q1" else q3
    if name == "size_bin":
        g = df[df["eval_set"] == args["eval_set"]]
        cut = np.where(g["gt_area"] < q1, "small", np.where(g["gt_area"] > q3, "large", "medium"))
        sub = g[cut == args["bin"]]
        per_seed = sub.groupby("seed")["dice"].mean()
        return {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1)), "n": float(sub["stem"].nunique())}[args["stat"]]
    raise KeyError(name)


def read_cell(root: Path, source: str, locator: str) -> str:
    """The text of the cell a locator points to."""
    kind, filt, column = _split(locator)
    path = root / source
    if kind == "csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if all(_same(r[k], v) for k, v in filt.items())]
        if len(rows) != 1:
            raise LookupError(f"{source} {locator}: {len(rows)} rows match")
        return rows[0][column]
    if kind == "json":
        node = json.loads(path.read_text(encoding="utf-8"))
        for part in column.split("."):
            node = node[part]
        return str(node)
    if kind == "md":
        col, _, idx = column.partition("@")
        rows = [r for r in _md_table(path.read_text(encoding="utf-8"), filt["section"]) if next(iter(r.values())) == filt["row"] and col in r]
        if len(rows) != 1:
            raise LookupError(f"{source} {locator}: {len(rows)} rows match")
        cell = rows[0][col]
        return cell.split("/")[int(idx)] if idx else cell
    if kind == "derived":
        name, _, rest = column.partition("(")
        args = dict(p.split("=", 1) for p in rest.rstrip(")").split(",") if p)
        return repr(_derived(root, source, name, args))
    raise ValueError(kind)


# ---------------------------------------------------------------- entries
def _e(rows: list, id_: str, meaning: str, source: str, locator: str, unit: str, decimals: int, derived: bool = False) -> None:
    rows.append({"id": id_, "meaning": meaning, "source_file": source, "locator": locator, "unit": unit, "decimals": decimals, "derived": derived})


def entries(root: Path = ROOT) -> List[dict]:
    rows: List[dict] = []
    runs = sorted(p.parent for p in (root / "runs_equal").glob("*/run.json"))
    for run in runs:
        name = run.name
        src = f"runs_equal/{name}/run.json"
        meta = json.loads((root / src).read_text(encoding="utf-8"))
        for key, unit, dec, text in (("seed", "", 0, "seed"), ("n_train", "images", 0, "training images"), ("n_val", "images", 0, "validation images"),
                                     ("epochs_run", "epochs", 0, "epochs run"), ("best_epoch", "epoch", 0, "best epoch (highest validation Dice)"),
                                     ("best_val_dice", "Dice", 3, "best validation Dice"), ("stop_reason", "text", 0, "stop reason")):
            _e(rows, f"run.{name}.{key}", f"{name}: {text}", src, f"json#{key}", unit, dec)
        for key in ("gpu", "device", "platform", "python", "torch", "numpy", "git_dirty"):
            if key in meta.get("environment", {}):
                _e(rows, f"run.{name}.env.{key}", f"{name}: training environment, {key}", src, f"json#environment.{key}", "text", 0)
        ev_src = f"runs_equal/{name}/eval_summary.json"
        own = f"{meta['source']}_test"
        ext = ("cvc" if meta["source"] == "kvasir" else "kvasir") + "_external"
        for tag, es in (("val", f"{meta['source']}_val"), ("own_test", own), ("external", ext)):
            _e(rows, f"run.{name}.{tag}.mean_dice", f"{name}: mean Dice on {es} (evaluation at the end of training)", ev_src, f"json#sets.{es}.mean_dice", "Dice", 3)
            _e(rows, f"run.{name}.{tag}.n_images", f"{name}: images in {es}", ev_src, f"json#sets.{es}.n_images", "images", 0)
    for cond, es in CONDITIONS.items():
        mbc = f"{TABLES}/metrics_by_condition.csv"
        for m in METRICS:
            dec = 1 if m == "hd95" else 3
            for col, text in (("mean_over_seeds", "mean over seeds"), ("sd_over_seeds", "sample SD over seeds"), ("median_over_images_mean_over_seeds", "mean over seeds of the per-seed median over images")):
                _e(rows, f"cond.{cond}.{m}.{col}", f"{cond} ({es}): {m}, {text}", mbc, f"csv?eval_set={es}&metric={m}#{col}", "px" if m == "hd95" else "fraction", dec)
        _e(rows, f"cond.{cond}.n_images", f"{cond} ({es}): images in the evaluation set", mbc, f"csv?eval_set={es}&metric=dice#n_images", "images", 0)
        _e(rows, f"cond.{cond}.n_seeds", f"{cond} ({es}): seeds", mbc, f"csv?eval_set={es}&metric=dice#n_seeds", "seeds", 0)
        for s in range(1, 6):
            _e(rows, f"cond.{cond}.dice.seed{s}", f"{cond} ({es}): mean Dice of seed {s}", mbc, f"csv?eval_set={es}&metric=dice#seed_{s}", "Dice", 3)
    for d in DIRECTIONS:
        tag = "kvasir" if d.startswith("Kvasir") else "cvc"
        gbs, gs = f"{TABLES}/gap_by_seed.csv", f"{TABLES}/gap_summary.csv"
        for s in range(1, 6):
            _e(rows, f"gap.{tag}.seed{s}", f"{d}: Dice gap of seed {s} (own test minus external)", gbs, f"csv?direction={d}&metric=dice&seed={s}#gap", "Dice", 3)
        for col, text in (("own_mean", "own-test Dice, mean over seeds"), ("external_mean", "external Dice, mean over seeds"), ("mean_gap", "mean gap"), ("sd_gap", "SD of the gap over seeds"),
                          ("t_crit", "t quantile (df = seeds - 1)"), ("ci_t_low", "t-based 95% interval, lower"), ("ci_t_high", "t-based 95% interval, upper"),
                          ("ci_boot_low", "cluster-bootstrap 95% interval, lower"), ("ci_boot_high", "cluster-bootstrap 95% interval, upper"), ("boot_samples", "bootstrap resamples")):
            _e(rows, f"gap.{tag}.{col}", f"{d}: Dice gap, {text}", gs, f"csv?direction={d}&metric=dice#{col}", "Dice", 3 if col != "boot_samples" else 0)
        for delta in ("0.05", "0.03", "0.10"):
            for col, text in (("decision_t_ci", "decision on the t-based interval"), ("decision_bootstrap_ci", "decision on the bootstrap interval")):
                _e(rows, f"decision.{tag}.d{delta}.{col}", f"{d}: {text}, delta = {delta}", f"{TABLES}/decision_rule.csv", f"csv?direction={d}&delta={delta}#{col}", "text", 0)
    sp = f"{TABLES}/spearman.csv"
    for es, var in (("cvc_external", "gt_area_fraction"), ("cvc_external", "brightness_gray"), ("kvasir_external", "gt_area_fraction"), ("kvasir_external", "brightness_gray")):
        for col, text, dec in (("spearman_rho", "Spearman rho", 3), ("n_images", "images", 0), ("p_value", "descriptive p-value", 6)):
            _e(rows, f"spearman.{es}.{var}.{col}", f"{es}: per-image Dice against {var}, {text}", sp, f"csv?eval_set={es}&variable={var}#{col}", "", dec)
    _e(rows, "errors.frames_below_0.5", "frames of the CVC-ClinicDB set with seed-averaged Dice < 0.5, Kvasir-SEG-trained model", PER_IMAGE, "derived#frames_below(thr=0.5)", "frames", 0, True)
    _e(rows, "errors.frames_total", "frames of the CVC-ClinicDB set", PER_IMAGE, "derived#frames_total()", "frames", 0, True)
    for which in ("q1", "q3"):
        _e(rows, f"size.cut.{which}", f"polyp-size bin limit {which} (quartile of the ground-truth area fraction over all evaluated images)", PER_IMAGE, f"derived#size_cut(which={which})", "fraction", 4, True)
    for cond, es in CONDITIONS.items():
        for b in ("small", "medium", "large"):
            for stat, text, dec in (("mean", "mean Dice over seeds", 3), ("sd", "sample SD over seeds", 3), ("n", "images", 0)):
                _e(rows, f"size.{cond}.{b}.{stat}", f"{cond} ({es}), {b} polyps: {text}", PER_IMAGE, f"derived#size_bin(eval_set={es},bin={b},stat={stat})", "images" if stat == "n" else "Dice", dec, True)
    for ds, key in (("Kvasir-SEG", "kvasir"), ("CVC-ClinicDB", "cvc")):
        for col, text in (("Matched pairs", "matched image-mask pairs"),):
            _e(rows, f"audit.{key}.pairs", f"{ds}: {text}", AUDIT, f"md?section=1.&row={ds}#{col}", "images", 0)
        _e(rows, f"audit.{key}.distinct_sizes", f"{ds}: distinct image sizes", AUDIT, f"md?section=4.&row={ds}#Distinct sizes", "sizes", 0)
        _e(rows, f"audit.{key}.masks_other_values", f"{ds}: masks with values other than 0 and 255", AUDIT, f"md?section=5.&row={ds}#Masks with other values", "masks", 0)
        _e(rows, f"audit.{key}.area_mean", f"{ds}: mean polyp area fraction", AUDIT, f"md?section=7.&row={ds}#Mean", "percent", 2)
        _e(rows, f"audit.{key}.grey_mean", f"{ds}: mean grey level", AUDIT, f"md?section=8.&row={ds}#Grey mean", "grey level", 2)
        for i, part in enumerate(("train", "val", "test")):
            _e(rows, f"split.{key}.{part}", f"{ds}: {part} pairs of the fixed split", AUDIT, f"md?section=11.&row={ds}#Train/val/test pairs@{i}", "images", 0)
    _e(rows, "audit.cvc.sequences", "CVC-ClinicDB: video sequences listed in metadata.csv", AUDIT, "md?section=11.&row=CVC-ClinicDB#Sequences", "sequences", 0)
    for scope, key in (("within cvc", "cvc"), ("within kvasir", "kvasir"), ("across datasets", "across")):
        _e(rows, f"audit.near_duplicates.{key}", f"near-duplicate pairs (perceptual hash distance <= 4), {scope}", AUDIT, f"md?section=10.&row={scope}#Near-duplicates (<= 4)", "pairs", 0)
    acc = f"{TABLES}/acceptance_checks.json"
    for es in ("kvasir_val", "kvasir_test", "cvc_external", "cvc_val", "cvc_test", "kvasir_external"):
        _e(rows, f"acceptance.A.{es}.max_abs_diff", f"recomputed against stored Dice, maximum absolute difference, {es}", acc, f"json#A_dice_reproduces_original.by_eval_set.{es}.max_abs_diff", "Dice", 4)
    _e(rows, "acceptance.B.max_abs_deviation", "IoU against Dice / (2 - Dice), maximum absolute deviation", acc, "json#B_iou_equals_dice_over_2_minus_dice.max_abs_deviation", "", 3)
    return rows


def build(root: Path = ROOT) -> pd.DataFrame:
    rows = entries(root)
    for r in rows:
        value = read_cell(root, r["source_file"], r["locator"])
        r["value"] = str(int(float(value))) if r["derived"] and r["decimals"] == 0 else value
    return pd.DataFrame(rows, columns=COLUMNS)


def to_markdown(df: pd.DataFrame) -> str:
    head = ["# Number ledger", "",
            f"{len(df)} values, each read from a committed file (`source_file`) at the place given by `locator`. `derived` = computed from the per-image file by the formulas of `train/readme_tables.py`. "
            "Verified by `python a4/check_ledger.py`.", "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    return "\n".join(head + ["| " + " | ".join(str(r[c]).replace("|", "/") for c in COLUMNS) + " |" for _, r in df.iterrows()]) + "\n"


def write(root: Path = ROOT, out: Optional[Path] = None) -> pd.DataFrame:
    out = out or root / "a4"
    out.mkdir(exist_ok=True)
    df = build(root)
    df.to_csv(out / "number_ledger.csv", index=False)
    (out / "number_ledger.md").write_text(to_markdown(df), encoding="utf-8")
    return df


if __name__ == "__main__":
    print(f"{len(write())} ledger rows written")
