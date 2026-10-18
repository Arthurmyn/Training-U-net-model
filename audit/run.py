"""Orchestration and CLI: ``python -m audit.run --config config.yaml``.

``run_audit`` computes everything in memory; ``write_outputs`` writes the files under ``outputs/``.
The notebook ``audit.ipynb`` uses the same two functions.
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import PIL
import scipy
import yaml

from . import checks, load, preprocess, report, splits
from .config import AuditConfig, DatasetConfig, load_config

log = logging.getLogger("audit")


@dataclass
class DatasetResult:
    """Everything computed for one dataset."""

    cfg: DatasetConfig
    files: load.Discovery
    table: pd.DataFrame
    summary: dict
    sequences: Dict[str, str] = field(default_factory=dict)
    sequence_note: str = ""
    sequence_info: dict = field(default_factory=dict)
    split: Optional[Dict[str, List[str]]] = None
    split_meta: dict = field(default_factory=dict)
    split_error: str = ""
    split_seq_of: Optional[Dict[str, str]] = None
    usable_stems: List[str] = field(default_factory=list)
    preprocess_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    preprocess_summary: dict = field(default_factory=dict)


@dataclass
class AuditResult:
    """Output of ``run_audit``."""

    config: AuditConfig
    datasets: Dict[str, DatasetResult]
    duplicates: dict
    leakage: dict
    environment: dict
    problems: List[str] = field(default_factory=list)
    flagged: pd.DataFrame = field(default_factory=pd.DataFrame)


def environment_info(seed: int) -> dict:
    """Python and library versions plus the seed (logged and stored in summary.json)."""
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "pillow": PIL.__version__, "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__, "seed": seed}


def log_environment(env: dict) -> None:
    """Emit the one-line environment record."""
    log.info("ENVIRONMENT " + " ".join(f"{k}={v}" for k, v in env.items()))


def build_split(res: DatasetResult, seed: int) -> None:
    """Create and validate the fixed split for one dataset (fills the ``split*`` fields of ``res``)."""
    ds = res.cfg
    stems = res.usable_stems
    excluded = sorted(set(res.files.matched) - set(stems), key=load.natural_key)
    meta: dict = {"seed": seed, "n_pairs": len(stems), "excluded_unreadable_pairs": excluded}
    try:
        if ds.split_method == "random":
            together = [[r.a, r.b] for r in ds.reviewed_pairs if r.verdict == "duplicate"]
            split = splits.random_split(stems, ds.split_sizes, seed, keep_together=together)
            meta.update(method="random", approximate=False, kept_together_pairs=together,
                        note="Random permutation of the sorted file stems with numpy default_rng(seed); "
                             "manually confirmed near-duplicate pairs are kept in the same split.")
            seq_of = None
        else:
            have_all = bool(res.sequences) and all(s in res.sequences for s in stems)
            if ds.split_method == "sequence" and have_all:
                seq_of = {s: res.sequences[s] for s in stems}
                split = splits.sequence_split(stems, seq_of, ds.split_fractions, seed)
                meta.update(method="sequence_grouped", approximate=False,
                            n_sequences=len(set(seq_of.values())), target_fractions=ds.split_fractions,
                            achieved_fractions={k: round(len(v) / len(stems), 4) for k, v in split.items()},
                            note="Whole video sequences are assigned to one split (seeded shuffle, largest first, "
                                 "greedy balancing). Fractions are matched as closely as whole sequences allow.")
            else:
                seq_of = None
                split = splits.block_split(stems, ds.split_fractions)
                why = res.sequence_note if not res.sequences else "sequence IDs missing for some images"
                meta.update(method="contiguous_blocks", approximate=True,
                            target_fractions=ds.split_fractions,
                            achieved_fractions={k: round(len(v) / len(stems), 4) for k, v in split.items()},
                            note=f"APPROXIMATION: no usable sequence IDs ({why}). Contiguous blocks of the numerically "
                                 "ordered files; frames of one video can still straddle a block boundary.")
        splits.validate_split(split, stems, seq_of)
        res.split, res.split_meta, res.split_seq_of = split, meta, seq_of
    except (ValueError, AssertionError, KeyError) as exc:
        res.split_error = f"{type(exc).__name__}: {exc}"
        res.split_meta = meta
        log.error("%s: split NOT created: %s", ds.name, res.split_error)


def sequence_integrity(sequences: Dict[str, str], split: Dict[str, List[str]], matched: List[str],
                       within_pairs: pd.DataFrame) -> dict:
    """Explicit check that no known video sequence crosses train/val/test.

    Also reports whether every sequence is a run of consecutive frame numbers (numeric stems only) and lists
    near-duplicate pairs that connect two *different* sequences (a hint that sequences may belong together).
    """
    where = {s: name for name, part in split.items() for s in part}
    seq_parts: Dict[str, set] = {}
    for stem, name in where.items():
        if stem in sequences:
            seq_parts.setdefault(sequences[stem], set()).add(name)
    crossing = sorted((q for q, parts in seq_parts.items() if len(parts) > 1), key=load.natural_key)
    contiguous = None
    if matched and all(m.isdigit() for m in matched):
        by_seq: Dict[str, List[int]] = {}
        for m in matched:
            if m in sequences:
                by_seq.setdefault(sequences[m], []).append(int(m))
        contiguous = all(max(v) - min(v) + 1 == len(v) for v in by_seq.values())
    cross_pairs = []
    for r in within_pairs.itertuples():
        a, b = Path(r.a_file).stem, Path(r.b_file).stem
        if a in sequences and b in sequences and sequences[a] != sequences[b]:
            cross_pairs.append({"a": r.a_file, "b": r.b_file, "hamming": int(r.hamming), "sequence_a": sequences[a],
                                "sequence_b": sequences[b], "split_a": where.get(a), "split_b": where.get(b)})
    return {"sequences_checked": len(seq_parts), "sequences_crossing_splits": crossing,
            "sequences_per_split": {name: sum(1 for parts in seq_parts.values() if name in parts) for name in split},
            "frame_ranges_contiguous": contiguous, "cross_sequence_near_duplicates": cross_pairs}


def split_leakage(result_ds: Dict[str, DatasetResult], near_pairs: pd.DataFrame) -> dict:
    """Count within-dataset near-duplicate pairs that fall into different splits.

    Pairs a person reviewed as ``different_images`` are counted separately; ``leaking_pairs`` is what remains.
    """
    out = {}
    for key, res in result_ds.items():
        if res.split is None:
            out[key] = None
            continue
        where = {res.files.images[s].name: name for name, part in res.split.items() for s in part}
        not_dup = {frozenset((r.a, r.b)) for r in res.cfg.reviewed_pairs if r.verdict == "different_images"}
        pairs = near_pairs[near_pairs["scope"] == f"within_{key}"]
        cross = [(a, b) for a, b in zip(pairs["a_file"], pairs["b_file"])
                 if a in where and b in where and where[a] != where[b]]
        leaking = [(a, b) for a, b in cross if frozenset((Path(a).stem, Path(b).stem)) not in not_dup]
        out[key] = {"near_duplicate_pairs": int(len(pairs)), "pairs_in_different_splits": len(cross),
                    "reviewed_as_different_images": len(cross) - len(leaking), "leaking_pairs": len(leaking),
                    "examples": [f"{a} ({where[a]}) ~ {b} ({where[b]})" for a, b in leaking[:5]]}
    return out


def run_audit(cfg: AuditConfig, progress: bool = True) -> AuditResult:
    """Run every check on every configured dataset and build the splits. Writes nothing."""
    env = environment_info(cfg.seed)
    log_environment(env)
    results: Dict[str, DatasetResult] = {}
    for key, ds in cfg.datasets.items():
        log.info("[%s] discovering files in %s", ds.name, ds.root.name)
        files = load.discover(ds)
        seqs, seq_note = load.load_sequence_ids(ds)
        table = checks.audit_dataset(cfg, ds, files, seqs, progress=progress)
        summary = checks.summarize_dataset(table, files, ds)
        res = DatasetResult(cfg=ds, files=files, table=table, summary=summary, sequences=seqs, sequence_note=seq_note)
        res.usable_stems = table.loc[table["image_ok"] & table["mask_ok"], "stem"].tolist()
        if seqs:
            used = [seqs[s] for s in files.matched if s in seqs]
            sizes = pd.Series(used).value_counts()
            res.sequence_info = {
                "source": seq_note, "n_sequences": int(sizes.size),
                "expected_sequences": ds.expected_sequences,
                "matches_expected": ds.expected_sequences is None or int(sizes.size) == ds.expected_sequences,
                "frames_per_sequence": {"min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max())},
                "pairs_with_sequence_id": len(used), "pairs_without_sequence_id": len(files.matched) - len(used)}
        else:
            res.sequence_info = {"source": seq_note, "n_sequences": None}
        build_split(res, cfg.seed)
        res.preprocess_table, res.preprocess_summary = preprocess.check_preprocessing(
            files, res.usable_stems, cfg.preprocess_size, cfg.mask_binarize)
        results[key] = res
        log.info("[%s] %d matched pairs, %d flagged files", ds.name, len(files.matched), summary["n_flagged_files"])

    dup = checks.find_duplicates({k: r.table for k, r in results.items()}, cfg.near_duplicate_hamming)
    leakage = split_leakage(results, dup["near_pairs"])
    for key, res in results.items():
        if res.sequences and res.split is not None:
            within = dup["near_pairs"][dup["near_pairs"]["scope"] == f"within_{key}"]
            res.sequence_info["integrity"] = sequence_integrity(res.sequences, res.split, res.files.matched, within)
    out = AuditResult(config=cfg, datasets=results, duplicates=dup, leakage=leakage, environment=env)
    out.flagged = report.flagged_files(out)
    out.problems = report.find_problems(out)
    return out


def write_outputs(result: AuditResult) -> List[Path]:
    """Write CSV, JSON, Markdown, split files and figures to the configured output folder."""
    out = result.config.output_dir
    (out / "splits").mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for key, res in result.datasets.items():
        p = out / f"per_image_stats_{key}.csv"
        res.table.to_csv(p, index=False, float_format="%.6f")
        written.append(p)
        p = out / f"image_sizes_{key}.csv"
        checks.size_table(res.table).to_csv(p, index=False)
        written.append(p)
        if res.split is not None:
            ext = res.usable_stems
            written += splits.write_split_files(out / "splits", key, res.cfg.name, res.split, res.split_meta,
                                                res.files.images, res.files.masks, res.split_seq_of, ext)
    p = out / "duplicate_pairs.csv"
    dp = result.duplicates["all_close_pairs"].copy()
    dp["near_duplicate"] = dp["hamming"] <= result.config.near_duplicate_hamming
    dp.to_csv(p, index=False)
    written.append(p)
    p = out / "flagged_files.csv"
    result.flagged.to_csv(p, index=False)
    written.append(p)
    written += report.make_figures(result, out)
    written += report.make_review_figure(result, out)
    written += report.make_preprocessing_figures(result, out)
    p = out / "summary.json"
    p.write_text(report.summary_json(result))
    written.append(p)
    p = out / "audit_report.md"
    p.write_text(report.render_markdown(result))
    written.append(p)
    for w in written:
        log.info("wrote %s", w.relative_to(result.config.base_dir))
    return written


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description="Dataset audit and fixed splits (Kvasir-SEG, CVC-ClinicDB).")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--no-progress", action="store_true", help="hide progress bars")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    result = run_audit(cfg, progress=not args.no_progress)
    write_outputs(result)
    for line in result.problems:
        log.warning("PROBLEM: %s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
