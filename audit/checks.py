"""Per-image checks, dataset summaries and duplicate detection. Nothing here writes or deletes files."""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.fft import dctn
from tqdm import tqdm

from .config import AuditConfig, DatasetConfig
from .load import Discovery, natural_key, open_image_rgb, read_mask_raw

log = logging.getLogger("audit")

PHASH_SIZE = 32   # image is shrunk to 32x32 grey, then the 8x8 low-frequency DCT block is used
PHASH_BITS = 64


# --------------------------------------------------------------------------- hashing
def file_md5(path) -> str:
    """MD5 of the raw file bytes (exact-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(rgb: Image.Image) -> str:
    """64-bit perceptual hash (DCT based, same recipe as imagehash.phash) as 16 hex characters."""
    grey = rgb.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.LANCZOS)
    dct = dctn(np.asarray(grey, dtype=np.float64), norm="ortho")[:8, :8]
    bits = (dct > np.median(dct)).flatten()
    return np.packbits(bits).tobytes().hex()


def hex_to_bits(hex_hashes: List[str]) -> np.ndarray:
    """Hex hashes -> (n, 64) uint8 bit matrix."""
    raw = np.frombuffer(bytes.fromhex("".join(hex_hashes)), dtype=np.uint8).reshape(len(hex_hashes), -1)
    return np.unpackbits(raw, axis=1)


# --------------------------------------------------------------------------- per image
def _error_text(exc: Exception, *paths) -> str:
    """Exception text with absolute paths replaced by file names (keeps outputs path-free)."""
    msg = f"{type(exc).__name__}: {exc}"
    for p in paths:
        msg = msg.replace(str(p), p.name).replace(p.as_posix(), p.name)
    return msg.replace("\n", " ")


def mask_stats(raw: np.ndarray, threshold: int, full_fraction: float) -> dict:
    """Statistics of one mask: raw values, binarization (> threshold), area, components."""
    unique = np.unique(raw)
    binary = raw > threshold
    area = float(binary.mean()) if binary.size else float("nan")
    _, n_comp = ndimage.label(binary)       # default structure = 4-connectivity
    values = {int(v) for v in unique}
    return {
        "mask_n_unique_values": int(unique.size),
        "mask_unique_values": ";".join(str(int(v)) for v in unique),
        "mask_noisy": bool(values - {0, 255}),
        "mask_n_classes_binarized": int(np.unique(binary).size),
        "polyp_area_fraction": area,
        "n_polyps": int(n_comp),
        "mask_empty": not bool(binary.any()),
        "mask_near_full": area > full_fraction,
    }


def audit_pair(stem: str, image_path, mask_path, cfg: AuditConfig,
               xcheck_image=None, xcheck_mask=None, sequence_id: Optional[str] = None) -> dict:
    """Run every per-image check for one image/mask pair. Never raises on bad files."""
    row: dict = {"stem": stem, "image_file": image_path.name, "mask_file": mask_path.name}
    if sequence_id is not None:
        row["sequence_id"] = sequence_id
    errors: List[str] = []
    rgb = None
    try:
        rgb = open_image_rgb(image_path)
        row.update(image_ok=True, img_width=rgb.width, img_height=rgb.height,
                   img_channels=rgb.info["orig_bands"], img_mode=rgb.info["orig_mode"])
        arr = np.asarray(rgb, dtype=np.float64)
        row["brightness_gray"] = float(np.asarray(rgb.convert("L"), dtype=np.float64).mean())
        row["mean_r"], row["mean_g"], row["mean_b"] = (float(arr[..., i].mean()) for i in range(3))
        row["phash"] = phash(rgb)
    except Exception as exc:  # corrupt / truncated / unsupported
        row["image_ok"] = False
        errors.append("image: " + _error_text(exc, image_path))
    row["image_md5"] = file_md5(image_path)

    raw = None
    try:
        raw, mode = read_mask_raw(mask_path)
        row.update(mask_ok=True, mask_width=raw.shape[1], mask_height=raw.shape[0], mask_mode=mode)
        row.update(mask_stats(raw, cfg.mask_binarize, cfg.full_mask_fraction))
    except Exception as exc:
        row["mask_ok"] = False
        errors.append("mask: " + _error_text(exc, mask_path))

    row["size_mismatch"] = bool(row["image_ok"] and row["mask_ok"] and
                                (row["img_width"], row["img_height"]) != (row["mask_width"], row["mask_height"]))
    if xcheck_image is not None and xcheck_mask is not None:
        row["crosscheck_image_identical"] = _same_pixels(rgb, xcheck_image, is_mask=False)
        row["crosscheck_mask_identical"] = _same_pixels(raw, xcheck_mask, is_mask=True)
    row["read_error"] = " | ".join(errors)
    return row


def _same_pixels(reference, other_path, is_mask: bool) -> Optional[bool]:
    """True/False if the second copy decodes to identical pixels; None if either is unreadable."""
    if reference is None:
        return None
    try:
        if is_mask:
            other, _ = read_mask_raw(other_path)
            return bool(np.array_equal(reference, other))
        return bool(np.array_equal(np.asarray(reference), np.asarray(open_image_rgb(other_path))))
    except Exception:
        return None


def audit_dataset(cfg: AuditConfig, ds: DatasetConfig, files: Discovery,
                  sequences: Dict[str, str], progress: bool = True) -> pd.DataFrame:
    """Per-image table for every matched pair, plus rows for unmatched files, with a ``flags`` column."""
    rows = []
    for stem in tqdm(files.matched, desc=ds.name, disable=not progress, mininterval=1.0):
        rows.append(audit_pair(
            stem, files.images[stem], files.masks[stem], cfg,
            xcheck_image=files.xcheck_images.get(stem), xcheck_mask=files.xcheck_masks.get(stem),
            sequence_id=sequences.get(stem) if sequences else None))
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds.name)
    return add_flags(df)


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a semicolon-separated ``flags`` column for problems that make a file noteworthy."""
    def flags(r) -> str:
        f = []
        if not r["image_ok"]:
            f.append("corrupt_image")
        if not r["mask_ok"]:
            f.append("corrupt_mask")
        if r["size_mismatch"]:
            f.append("size_mismatch")
        if r.get("mask_ok") is True:
            if r["mask_empty"]:
                f.append("empty_mask")
            if r["mask_near_full"]:
                f.append("near_full_mask")
            if r["mask_n_classes_binarized"] != 2 and not r["mask_empty"] and not r["mask_near_full"]:
                f.append("not_two_class")
        if r.get("crosscheck_image_identical") is False:
            f.append("png_tif_image_differs")
        if r.get("crosscheck_mask_identical") is False:
            f.append("png_tif_mask_differs")
        return ";".join(f)

    out = df.copy()
    out["flags"] = out.apply(flags, axis=1) if len(out) else ""
    return out


# --------------------------------------------------------------------------- summaries
def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {k: None for k in ("mean", "std", "min", "q25", "median", "q75", "max")}
    return {"mean": float(s.mean()), "std": float(s.std(ddof=0)), "min": float(s.min()),
            "q25": float(s.quantile(0.25)), "median": float(s.median()),
            "q75": float(s.quantile(0.75)), "max": float(s.max())}


def size_table(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct (width, height, channels) with counts, most frequent first."""
    ok = df[df["image_ok"]]
    t = (ok.groupby(["img_width", "img_height", "img_channels"]).size().reset_index(name="count")
         .sort_values(["count", "img_width", "img_height"], ascending=[False, True, True]).reset_index(drop=True))
    return t


def summarize_dataset(df: pd.DataFrame, files: Discovery, ds: DatasetConfig) -> dict:
    """Numeric summary of one dataset (becomes part of summary.json and the report)."""
    ok_img = df[df["image_ok"]]
    ok_mask = df[df["mask_ok"]]
    sizes = size_table(df)
    s = {
        "name": ds.name,
        "counts": {
            "images": len(files.images), "masks": len(files.masks), "matched_pairs": len(files.matched),
            "unmatched_images": len(files.unmatched_images), "unmatched_masks": len(files.unmatched_masks),
            "expected_pairs": ds.expected_count,
            "matches_expected": len(files.matched) == ds.expected_count and not files.unmatched_images
                                and not files.unmatched_masks,
            "unmatched_image_stems": files.unmatched_images, "unmatched_mask_stems": files.unmatched_masks,
            "ambiguous_stems": files.ambiguous_stems,
        },
        "unreadable": {
            "corrupt_images": df.loc[~df["image_ok"], "image_file"].tolist(),
            "corrupt_masks": df.loc[~df["mask_ok"], "mask_file"].tolist(),
        },
        "sizes": {
            "distinct": int(len(sizes)),
            "width": _stats(ok_img["img_width"]), "height": _stats(ok_img["img_height"]),
            "channels": {str(k): int(v) for k, v in ok_img["img_channels"].value_counts().sort_index().items()},
            "n_size_mismatch": int(df["size_mismatch"].sum()),
        },
        "mask_values": {
            "mask_modes": {str(k): int(v) for k, v in ok_mask["mask_mode"].value_counts().sort_index().items()},
            "n_masks_only_0_255": int((~ok_mask["mask_noisy"]).sum()),
            "n_masks_other_values": int(ok_mask["mask_noisy"].sum()),
            "max_distinct_values_in_one_mask": int(ok_mask["mask_n_unique_values"].max()) if len(ok_mask) else 0,
            "n_binarized_two_class": int((ok_mask["mask_n_classes_binarized"] == 2).sum()),
            "n_binarized_one_class": int((ok_mask["mask_n_classes_binarized"] == 1).sum()),
            "threshold": "value > 127",
        },
        "empty_masks": ok_mask.loc[ok_mask["mask_empty"], "stem"].tolist(),
        "near_full_masks": ok_mask.loc[ok_mask["mask_near_full"], "stem"].tolist(),
        "area_fraction": _stats(ok_mask["polyp_area_fraction"]),
        "n_polyps": {
            **_stats(ok_mask["n_polyps"]),
            "histogram": {str(k): int(v) for k, v in ok_mask["n_polyps"].clip(upper=4).value_counts().sort_index().items()},
            "histogram_note": "key 4 means 4 or more",
        },
        "brightness_gray": _stats(ok_img["brightness_gray"]),
        "channel_means": {c: _stats(ok_img[f"mean_{c}"]) for c in ("r", "g", "b")},
        "n_flagged_files": int((df["flags"] != "").sum()),
    }
    if "crosscheck_image_identical" in df.columns:
        has_img = int(df["stem"].isin(files.xcheck_images).sum())
        has_mask = int(df["stem"].isin(files.xcheck_masks).sum())
        s["crosscheck"] = {
            "images_second_copy_found": has_img,
            "images_compared": int(df["crosscheck_image_identical"].notna().sum()),
            "images_identical": int((df["crosscheck_image_identical"] == True).sum()),  # noqa: E712
            "images_second_copy_undecodable": has_img - int(df["crosscheck_image_identical"].notna().sum()),
            "masks_second_copy_found": has_mask,
            "masks_compared": int(df["crosscheck_mask_identical"].notna().sum()),
            "masks_identical": int((df["crosscheck_mask_identical"] == True).sum()),  # noqa: E712
        }
    return s


# --------------------------------------------------------------------------- manual review support
def pair_similarity(image_a, image_b, mask_a, mask_b, threshold: int = 127, size: int = 128) -> dict:
    """Numbers that support a visual near-duplicate review: grey-image correlation and mask IoU (both at 128x128)."""
    def grey(p):
        return np.asarray(open_image_rgb(p).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64)

    def binary(p):
        raw, _ = read_mask_raw(p)
        return np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((size, size), Image.NEAREST)) > threshold

    ga, gb = grey(image_a), grey(image_b)
    ma, mb = binary(mask_a), binary(mask_b)
    union = int((ma | mb).sum())
    return {"image_correlation": float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1]),
            "image_mean_abs_diff": float(np.abs(ga - gb).mean()),
            "mask_iou": float((ma & mb).sum() / union) if union else float("nan")}


# --------------------------------------------------------------------------- duplicates
def find_duplicates(frames: Dict[str, pd.DataFrame], max_hamming: int) -> dict:
    """Exact (MD5) and near (pHash Hamming <= ``max_hamming``) duplicates, within and across datasets.

    Report only: nothing is removed. ``frames`` maps dataset key -> per-image table.
    Returns dict with ``exact_groups`` (list), ``near_pairs`` (DataFrame) and ``distance_histogram``.
    """
    parts = []
    for key, df in frames.items():
        d = df[df["image_ok"]][["dataset", "stem", "image_file", "image_md5", "phash"]].copy()
        d["key"] = key
        if "sequence_id" in df.columns:
            d["sequence_id"] = df.loc[d.index, "sequence_id"]
        parts.append(d)
    allimg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    exact_groups = []
    for md5, g in allimg.groupby("image_md5"):
        if len(g) > 1:
            exact_groups.append({"md5": md5, "files": [f"{r.key}/{r.image_file}" for r in g.itertuples()]})
    exact_groups.sort(key=lambda g: natural_key(g["files"][0]))

    bits = hex_to_bits(allimg["phash"].tolist()).astype(np.float32)
    agree = bits @ bits.T + (1 - bits) @ (1 - bits).T
    dist = (PHASH_BITS - agree).astype(np.int16)
    iu, ju = np.where(np.triu(dist <= max(max_hamming, 8), k=1))
    pairs = []
    for i, j in zip(iu, ju):
        a, b = allimg.iloc[i], allimg.iloc[j]
        same_ds = a["key"] == b["key"]
        pairs.append({
            "scope": f"within_{a['key']}" if same_ds else "across_datasets",
            "a_dataset": a["key"], "a_file": a["image_file"], "b_dataset": b["key"], "b_file": b["image_file"],
            "hamming": int(dist[i, j]), "exact_bytes": bool(a["image_md5"] == b["image_md5"]),
            "same_sequence": (bool(a["sequence_id"] == b["sequence_id"]) if same_ds and "sequence_id" in allimg
                              and pd.notna(a.get("sequence_id")) else None),
        })
    cols = ["scope", "a_dataset", "a_file", "b_dataset", "b_file", "hamming", "exact_bytes", "same_sequence"]
    pairs.sort(key=lambda p: (p["scope"], p["hamming"], natural_key(p["a_file"]), natural_key(p["b_file"])))
    pairs_df = pd.DataFrame(pairs, columns=cols)
    hist = {}
    for scope, g in pairs_df.groupby("scope"):
        hist[scope] = {int(k): int(v) for k, v in g["hamming"].value_counts().sort_index().items()}
    near = pairs_df[pairs_df["hamming"] <= max_hamming].reset_index(drop=True)
    return {"exact_groups": exact_groups, "near_pairs": near, "all_close_pairs": pairs_df, "distance_histogram": hist}
