"""Deterministic train/val/test splits and the checks that they are valid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .load import natural_key

SPLIT_NAMES = ("train", "val", "test")


def random_split(stems: Sequence[str], sizes: Mapping[str, int], seed: int) -> Dict[str, List[str]]:
    """Seeded random split with exact sizes. Input order does not matter (stems are sorted first)."""
    ordered = sorted(stems, key=natural_key)
    if sum(sizes.values()) != len(ordered):
        raise ValueError(f"split sizes {dict(sizes)} sum to {sum(sizes.values())}, but {len(ordered)} pairs are available")
    perm = np.random.default_rng(seed).permutation(len(ordered))
    out: Dict[str, List[str]] = {}
    start = 0
    for name in SPLIT_NAMES:
        n = int(sizes[name])
        out[name] = sorted((ordered[i] for i in perm[start:start + n]), key=natural_key)
        start += n
    return out


def sequence_split(stems: Sequence[str], seq_of: Mapping[str, str], fractions: Mapping[str, float],
                   seed: int) -> Dict[str, List[str]]:
    """Assign whole video sequences to splits so that no sequence is shared between splits.

    Sequences are shuffled (seeded), sorted largest first (stable) and each goes to the split that is
    furthest below its target number of frames. The result approximates ``fractions`` as closely as
    whole sequences allow.
    """
    groups: Dict[str, List[str]] = {}
    for s in sorted(stems, key=natural_key):
        groups.setdefault(str(seq_of[s]), []).append(s)
    seq_ids = sorted(groups, key=natural_key)
    order = [seq_ids[i] for i in np.random.default_rng(seed).permutation(len(seq_ids))]
    order.sort(key=lambda q: -len(groups[q]))           # stable: ties keep the shuffled order
    n = len(stems)
    target = {k: fractions[k] * n for k in SPLIT_NAMES}
    count = {k: 0 for k in SPLIT_NAMES}
    out: Dict[str, List[str]] = {k: [] for k in SPLIT_NAMES}
    for q in order:
        best = max(SPLIT_NAMES, key=lambda k: (target[k] - count[k], -SPLIT_NAMES.index(k)))
        out[best].extend(groups[q])
        count[best] += len(groups[q])
    return {k: sorted(v, key=natural_key) for k, v in out.items()}


def block_split(stems: Sequence[str], fractions: Mapping[str, float]) -> Dict[str, List[str]]:
    """Contiguous blocks of the (numerically) ordered file list. Approximation used without sequence IDs."""
    ordered = sorted(stems, key=natural_key)
    n = len(ordered)
    n_train = int(round(fractions["train"] * n))
    n_val = int(round(fractions["val"] * n))
    return {"train": ordered[:n_train], "val": ordered[n_train:n_train + n_val], "test": ordered[n_train + n_val:]}


def validate_split(split: Mapping[str, Sequence[str]], all_stems: Sequence[str],
                   seq_of: Optional[Mapping[str, str]] = None) -> None:
    """Assert splits are disjoint, cover every pair exactly once, and (if given) never share a sequence."""
    flat = [s for part in split.values() for s in part]
    assert len(flat) == len(set(flat)), "a pair appears more than once across/within splits"
    assert set(flat) == set(all_stems), "splits do not cover every pair exactly once"
    assert len(flat) == len(all_stems), "split size differs from number of pairs"
    if seq_of is not None:
        owner: Dict[str, str] = {}
        for name, part in split.items():
            for s in part:
                prev = owner.setdefault(str(seq_of[s]), name)
                assert prev == name, f"sequence {seq_of[s]} appears in both '{prev}' and '{name}'"


def write_split_files(out_dir: Path, key: str, name: str, split: Mapping[str, Sequence[str]],
                      meta: dict, images: Mapping[str, Path], masks: Mapping[str, Path],
                      seq_of: Optional[Mapping[str, str]], external: Sequence[str]) -> List[Path]:
    """Write ``<key>_split.json``, ``<key>_split.csv`` and ``<key>_external_test.json`` (deterministic bytes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    body = {"dataset": name, **meta,
            "sizes": {k: len(v) for k, v in split.items()},
            "splits": {k: [images[s].name for s in v] for k, v in split.items()}}
    p = out_dir / f"{key}_split.json"
    p.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    written.append(p)

    lines = ["image_file,mask_file,split" + (",sequence_id" if seq_of else "")]
    for k in SPLIT_NAMES:
        for s in split[k]:
            lines.append(f"{images[s].name},{masks[s].name},{k}" + (f",{seq_of[s]}" if seq_of else ""))
    p = out_dir / f"{key}_split.csv"
    p.write_text("\n".join(lines) + "\n")
    written.append(p)

    ext = {"dataset": name, "purpose": "whole dataset used as external test set for a model trained on the other dataset",
           "n_pairs": len(external), "images": [images[s].name for s in external],
           "masks": [masks[s].name for s in external]}
    p = out_dir / f"{key}_external_test.json"
    p.write_text(json.dumps(ext, indent=2, sort_keys=True) + "\n")
    written.append(p)
    return written
