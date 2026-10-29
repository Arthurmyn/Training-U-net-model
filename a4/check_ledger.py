"""Re-read every cell named in a4/number_ledger.csv and fail on any mismatch: ``python a4/check_ledger.py``.

Independent of train/a4_ledger.py: the locators are implemented again here with plain csv/json/re code.
Exit code 0: all values match their sources; 1: at least one mismatch or unreadable source.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _equal(a: str, b: str) -> bool:
    x, y = _num(a), _num(b)
    return abs(x - y) <= 1e-9 * max(1.0, abs(x)) if x is not None and y is not None else a == b


def _quantile(values, q):
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _derived(root: Path, source: str, call: str) -> float:
    with open(root / source, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    name, args = re.match(r"(\w+)\((.*)\)", call).groups()
    args = dict(p.split("=", 1) for p in args.split(",") if p)
    if name in ("frames_below", "frames_total"):
        per = defaultdict(list)
        for r in rows:
            if r["source"] == "kvasir" and r["eval_set"] == "cvc_external":
                per[r["stem"]].append(float(r["dice"]))
        if name == "frames_total":
            return float(len(per))
        return float(sum(sum(v) / len(v) < float(args["thr"]) for v in per.values()))
    seen = {}
    for r in rows:
        seen.setdefault((r["eval_set"], r["stem"]), float(r["gt_area"]))
    q1, q3 = _quantile(list(seen.values()), 0.25), _quantile(list(seen.values()), 0.75)
    if name == "size_cut":
        return q1 if args["which"] == "q1" else q3
    label = lambda a: "small" if a < q1 else ("large" if a > q3 else "medium")  # noqa: E731
    per_seed, stems = defaultdict(list), set()
    for r in rows:
        if r["eval_set"] == args["eval_set"] and label(float(r["gt_area"])) == args["bin"]:
            per_seed[r["seed"]].append(float(r["dice"]))
            stems.add(r["stem"])
    means = [sum(v) / len(v) for v in per_seed.values()]
    return {"mean": statistics.fmean(means), "sd": statistics.stdev(means), "n": float(len(stems))}[args["stat"]]


def _md_cell(text: str, section: str, row: str, column: str) -> str:
    column, _, idx = column.partition("@")
    block, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.startswith("## " + section)
        if inside:
            block.append(line)
    header = None
    for line in block:
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[-: ]*", c) for c in cells):
            continue
        if header is None:
            header = cells
        elif cells[0] == row and column in header:
            cell = cells[header.index(column)]
            return cell.split("/")[int(idx)] if idx else cell
    raise LookupError(f"no row {row!r} with column {column!r} in section {section!r}")


def reread(root: Path, source: str, locator: str) -> str:
    head, _, tail = locator.partition("#")
    kind, _, query = head.partition("?")
    q = dict(p.split("=", 1) for p in query.split("&") if p)
    path = root / source
    if kind == "csv":
        with open(path, newline="", encoding="utf-8") as f:
            hits = [r for r in csv.DictReader(f) if all(r[k] == v or (_num(r[k]) is not None and _num(v) is not None and float(r[k]) == float(v)) for k, v in q.items())]
        if len(hits) != 1:
            raise LookupError(f"{len(hits)} rows match")
        return hits[0][tail]
    if kind == "json":
        node = json.loads(path.read_text(encoding="utf-8"))
        for part in tail.split("."):
            node = node[part]
        return str(node)
    if kind == "md":
        return _md_cell(path.read_text(encoding="utf-8"), q["section"], q["row"], tail)
    if kind == "derived":
        return repr(_derived(root, source, tail))
    raise ValueError(f"unknown locator kind {kind!r}")


def check(root: Path = ROOT, ledger: Path = None) -> list:
    """List of problems (empty when every ledger value equals its source cell)."""
    ledger = ledger or root / "a4" / "number_ledger.csv"
    with open(ledger, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    problems, seen = [], set()
    for r in rows:
        if r["id"] in seen:
            problems.append(f"{r['id']}: duplicate id")
        seen.add(r["id"])
        try:
            actual = reread(root, r["source_file"], r["locator"])
        except Exception as exc:  # noqa: BLE001 - any unreadable source is a failure
            problems.append(f"{r['id']}: cannot read {r['source_file']} {r['locator']}: {exc}")
            continue
        if not _equal(r["value"], actual):
            problems.append(f"{r['id']}: ledger {r['value']} but source {actual} ({r['source_file']} {r['locator']})")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"{len(problems)} problem(s):")
        print("\n".join(problems))
        return 1
    with open(ROOT / "a4" / "number_ledger.csv", newline="", encoding="utf-8") as f:
        n = sum(1 for _ in csv.DictReader(f))
    print(f"ledger OK: {n} values match their source cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
