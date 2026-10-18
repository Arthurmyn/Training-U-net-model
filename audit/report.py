"""Figures, summary.json and the Markdown report. Output text is deterministic (no timestamps)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .checks import size_table  # noqa: E402
from .load import natural_key  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .run import AuditResult

# Validated categorical slots 1-2 (blue, orange) on a light surface; text uses ink tokens.
COLOR_CYCLE = ["#2a78d6", "#eb6834"]
MARKER_CYCLE = ["o", "^"]


def _color(result, key: str) -> str:
    """Fixed colour per dataset, by its position in the config (never cycled past two datasets)."""
    return COLOR_CYCLE[list(result.datasets).index(key) % len(COLOR_CYCLE)]


def _marker(result, key: str) -> str:
    return MARKER_CYCLE[list(result.datasets).index(key) % len(MARKER_CYCLE)]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MAX_LIST = 10


# --------------------------------------------------------------------------- helpers
def _clean(o: Any) -> Any:
    """Make an object JSON-safe (numpy scalars, NaN -> None, paths -> str)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return o.name
    return o


def _f(x, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _pct(x, nd: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    """GitHub-flavoured Markdown table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- findings
def flagged_files(result: "AuditResult") -> pd.DataFrame:
    """Every flagged file: per-image flags, unmatched files, exact duplicates, cross-dataset near-duplicates."""
    rows = []
    for key, res in result.datasets.items():
        name = res.cfg.name
        for r in res.table[res.table["flags"] != ""].itertuples():
            for flag in r.flags.split(";"):
                rows.append((name, r.image_file, flag, r.read_error if "corrupt" in flag else ""))
        for s in res.files.unmatched_images:
            rows.append((name, res.files.images[s].name, "unmatched_image", "no mask with this file name stem"))
        for s in res.files.unmatched_masks:
            rows.append((name, res.files.masks[s].name, "unmatched_mask", "no image with this file name stem"))
        for s in res.files.ambiguous_stems:
            rows.append((name, s, "ambiguous_stem", "several files share this stem; first one used"))
    for g in result.duplicates["exact_groups"]:
        for f in g["files"]:
            ds_key, fname = f.split("/", 1)
            rows.append((result.datasets[ds_key].cfg.name, fname, "exact_duplicate", "same bytes as: " + ", ".join(x for x in g["files"] if x != f)))
    near = result.duplicates["near_pairs"]
    for r in near[near["scope"] == "across_datasets"].itertuples():
        for ds_key, fname, other in ((r.a_dataset, r.a_file, f"{r.b_dataset}/{r.b_file}"), (r.b_dataset, r.b_file, f"{r.a_dataset}/{r.a_file}")):
            rows.append((result.datasets[ds_key].cfg.name, fname, "near_duplicate_across_datasets", f"hamming {r.hamming} to {other}"))
    rows.sort(key=lambda r: (r[0], natural_key(r[1]), r[2]))
    return pd.DataFrame(rows, columns=["dataset", "file", "flag", "detail"])


def find_problems(result: "AuditResult") -> List[str]:
    """Plain-English list of everything that deviates from expectations."""
    p: List[str] = []
    for key, res in result.datasets.items():
        s, n, c = res.summary, res.cfg.name, res.summary["counts"]
        if c["matched_pairs"] != c["expected_pairs"]:
            p.append(f"{n}: found {c['matched_pairs']} matched pairs, expected {c['expected_pairs']}.")
        if c["unmatched_images"] or c["unmatched_masks"]:
            p.append(f"{n}: {c['unmatched_images']} image(s) without mask and {c['unmatched_masks']} mask(s) without image.")
        if c["ambiguous_stems"]:
            p.append(f"{n}: {len(c['ambiguous_stems'])} file name stem(s) occur with several extensions.")
        bad = len(s["unreadable"]["corrupt_images"]) + len(s["unreadable"]["corrupt_masks"])
        if bad:
            p.append(f"{n}: {bad} unreadable/corrupt file(s) (excluded from splits).")
        if s["sizes"]["n_size_mismatch"]:
            p.append(f"{n}: {s['sizes']['n_size_mismatch']} image/mask pair(s) differ in size.")
        mv = s["mask_values"]
        if mv["n_masks_other_values"]:
            p.append(f"{n}: {mv['n_masks_other_values']} of {c['masks']} masks contain values other than 0/255 "
                     f"(up to {mv['max_distinct_values_in_one_mask']} distinct values in one mask; compression or anti-aliasing "
                     f"artefacts). They are binarised with > {result.config.mask_binarize}; "
                     f"{mv['n_binarized_two_class']} of {len(res.table)} binarised masks have exactly two classes.")
        if s["empty_masks"]:
            p.append(f"{n}: {len(s['empty_masks'])} empty mask(s).")
        if s["near_full_masks"]:
            p.append(f"{n}: {len(s['near_full_masks'])} mask(s) cover more than {_pct(result.config.full_mask_fraction, 0)}% of the image.")
        si = res.sequence_info
        if si.get("n_sequences") and not si.get("matches_expected", True):
            p.append(f"{n}: the sequence metadata lists {si['n_sequences']} video sequences, but {si['expected_sequences']} were expected.")
        if si.get("pairs_without_sequence_id"):
            p.append(f"{n}: {si['pairs_without_sequence_id']} pair(s) have no sequence ID.")
        x = s.get("crosscheck")
        if x and x["images_second_copy_undecodable"]:
            p.append(f"{n}: the second (TIFF) copy of {x['images_second_copy_undecodable']} images could not be decoded by Pillow, "
                     "so the PNG/TIFF image cross-check was not possible (masks: "
                     f"{x['masks_identical']}/{x['masks_compared']} identical).")
        if res.split_error:
            p.append(f"{n}: split NOT created ({res.split_error}).")
        elif res.split_meta.get("approximate"):
            p.append(f"{n}: split is an approximation (contiguous blocks, no sequence IDs).")
    d = result.duplicates
    if d["exact_groups"]:
        p.append(f"{len(d['exact_groups'])} group(s) of byte-identical images.")
    near = d["near_pairs"]
    for scope, g in near.groupby("scope"):
        extra = ""
        if scope.startswith("within_") and "same_sequence" in g and g["same_sequence"].notna().any():
            extra = f" ({int((g['same_sequence'] == True).sum())} of them within the same video sequence)"  # noqa: E712
        label = "between the two datasets" if scope == "across_datasets" else f"within {result.datasets[scope[7:]].cfg.name}"
        p.append(f"{len(g)} near-duplicate image pair(s) {label} (pHash Hamming <= {result.config.near_duplicate_hamming}){extra}.")
    for key, lk in result.leakage.items():
        if lk and lk["leaking_pairs"]:
            p.append(f"{result.datasets[key].cfg.name}: {lk['leaking_pairs']} near-duplicate pair(s) end up in different splits (possible leakage).")
    return p


# --------------------------------------------------------------------------- figures
def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, path: Path) -> Path:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return path


def _overlay_hist(result, column: str, scale: float, bins, title: str, xlabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, res in result.datasets.items():
        v = res.table.loc[res.table["image_ok"] & res.table["mask_ok"], column].dropna() * scale
        n = len(v)
        ax.hist(v, bins=bins, density=True, color=_color(result, key), alpha=0.40, linewidth=0)
        ax.hist(v, bins=bins, density=True, histtype="step", color=_color(result, key), linewidth=2,
                label=f"{res.cfg.name} (n={n}, median {v.median():.1f})")
    _style(ax, title, xlabel, "Density (area under each curve = 1)")
    leg = ax.legend(frameon=False, labelcolor=INK)
    return _save(fig, path)


def make_figures(result: "AuditResult", out_dir: Path) -> List[Path]:
    """Write the three PNG figures."""
    paths = [
        _overlay_hist(result, "polyp_area_fraction", 100, np.linspace(0, 90, 46),
                      "Polyp area fraction per image", "Polyp area (% of image)", out_dir / "fig_polyp_area_fraction.png"),
        _overlay_hist(result, "brightness_gray", 1, np.linspace(0, 160, 41),
                      "Mean image brightness (grayscale)", "Mean grey level (0-255)", out_dir / "fig_brightness_hist.png"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key, res in result.datasets.items():
        t = size_table(res.table)
        size = 90 if len(t) <= 3 else 28
        ax.scatter(t["img_width"], t["img_height"], s=size, c=_color(result, key), marker=_marker(result, key), alpha=0.75,
                   edgecolors=SURFACE, linewidths=1.2,
                   label=f"{res.cfg.name} ({len(t)} distinct size{'s' if len(t) != 1 else ''}, {int(t['count'].sum())} images)")
    _style(ax, "Image sizes", "Width (px)", "Height (px)")
    ax.legend(frameon=False, labelcolor=INK)
    paths.insert(1, _save(fig, out_dir / "fig_image_sizes.png"))
    return paths


# --------------------------------------------------------------------------- summary.json
def summary_json(result: "AuditResult") -> str:
    """Machine-readable summary (deterministic text)."""
    dup = result.duplicates
    body = {
        "environment": result.environment,
        "settings": result.config.settings_dict(),
        "problems": result.problems,
        "datasets": {},
        "duplicates": {
            "exact_groups": dup["exact_groups"],
            "near_pairs_by_scope": {k: int(v) for k, v in dup["near_pairs"]["scope"].value_counts().sort_index().items()},
            "hamming_histogram_by_scope": dup["distance_histogram"],
            "method": f"pHash 64 bit (32x32 grey, 8x8 DCT, median threshold); near-duplicate = Hamming <= {result.config.near_duplicate_hamming}",
        },
        "split_leakage_check": result.leakage,
        "n_flagged_file_entries": int(len(result.flagged)),
    }
    for key, res in result.datasets.items():
        body["datasets"][key] = {
            **res.summary,
            "sequence_info": res.sequence_info,
            "split": {**res.split_meta, "error": res.split_error or None,
                      "sizes": {k: len(v) for k, v in (res.split or {}).items()}},
        }
    return json.dumps(_clean(body), indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------- markdown
def render_markdown(result: "AuditResult") -> str:
    """The audit report, written so tables and sentences can be pasted into a methodology section."""
    cfg, env = result.config, result.environment
    ds = result.datasets
    L: List[str] = ["# Dataset audit report", ""]
    L.append(f"Seed {env['seed']}; Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scipy {env['scipy']}, "
             f"Pillow {env['pillow']}, matplotlib {env['matplotlib']}. Masks are binarised with value > {cfg.mask_binarize}; "
             f"a mask is 'near-full' above {_pct(cfg.full_mask_fraction, 0)}% polyp area; near-duplicates are images whose 64-bit "
             f"perceptual hash differs in at most {cfg.near_duplicate_hamming} bits.")
    L.append("")

    L += ["## 1. Pair counts", ""]
    rows = []
    for r in ds.values():
        c = r.summary["counts"]
        rows.append([r.cfg.name, c["images"], c["masks"], c["matched_pairs"], c["unmatched_images"], c["unmatched_masks"],
                     c["expected_pairs"], "OK" if c["matches_expected"] else "MISMATCH"])
    L += [md_table(["Dataset", "Images", "Masks", "Matched pairs", "Unmatched images", "Unmatched masks", "Expected", "Status"], rows), ""]
    for r in ds.values():
        c = r.summary["counts"]
        if c["matches_expected"]:
            L.append(f"- **{r.cfg.name}:** {c['matched_pairs']} matched image-mask pairs were found, as expected ({c['expected_pairs']}); no unmatched files.")
        else:
            L.append(f"- **{r.cfg.name}:** {c['matched_pairs']} matched image-mask pairs were found, but {c['expected_pairs']} were expected "
                     f"({c['unmatched_images']} unmatched image(s), {c['unmatched_masks']} unmatched mask(s)).")
    L.append("")

    L += ["## 2. Problems found", ""]
    L += [f"- {p}" for p in result.problems] or ["- None."]
    L.append("")

    L += ["## 3. Unreadable files", ""]
    for r in ds.values():
        u = r.summary["unreadable"]
        if u["corrupt_images"] or u["corrupt_masks"]:
            L.append(f"- **{r.cfg.name}:** unreadable images: {', '.join(u['corrupt_images']) or 'none'}; unreadable masks: {', '.join(u['corrupt_masks']) or 'none'}.")
        else:
            L.append(f"- **{r.cfg.name}:** all {len(r.table)} images and all masks opened and decoded completely.")
    L.append("")

    L += ["## 4. Image sizes", ""]
    rows = []
    for r in ds.values():
        z = r.summary["sizes"]
        ch = ", ".join(f"{k} ch: {v}" for k, v in z["channels"].items())
        rows.append([r.cfg.name, z["distinct"], f"{int(z['width']['min'])}-{int(z['width']['max'])} (median {int(z['width']['median'])})",
                     f"{int(z['height']['min'])}-{int(z['height']['max'])} (median {int(z['height']['median'])})", ch, z["n_size_mismatch"]])
    L += [md_table(["Dataset", "Distinct sizes", "Width px", "Height px", "Channels", "Image/mask size mismatches"], rows), ""]
    for r in ds.values():
        t = size_table(r.table)
        L += [f"Most frequent sizes, {r.cfg.name} (full table: `image_sizes_{r.cfg.key}.csv`):", ""]
        top = t.head(MAX_LIST)
        L.append(md_table(["Width", "Height", "Channels", "Images"], [[x.img_width, x.img_height, x.img_channels, x.count] for x in top.itertuples()]))
        if len(t) > MAX_LIST:
            L.append(f"\n... and {len(t) - MAX_LIST} more distinct sizes covering {int(t['count'].iloc[MAX_LIST:].sum())} images.")
        L.append("")

    L += ["## 5. Mask values", ""]
    rows = []
    for r in ds.values():
        m = r.summary["mask_values"]
        modes = ", ".join(f"{k}: {v}" for k, v in m["mask_modes"].items())
        rows.append([r.cfg.name, modes, m["n_masks_only_0_255"], m["n_masks_other_values"], m["max_distinct_values_in_one_mask"],
                     m["n_binarized_two_class"], m["n_binarized_one_class"]])
    L += [md_table(["Dataset", "Mask file mode", "Masks with only {0,255}", "Masks with other values", "Max distinct values in one mask",
                    "Binarised: 2 classes", "Binarised: 1 class"], rows), "",
          "Unique raw values of every mask are stored in the column `mask_unique_values` of the per-image CSV. "
          "Multi-channel masks are reduced to one channel with the maximum over channels before thresholding.", ""]

    L += ["## 6. Empty and near-full masks", ""]
    for r in ds.values():
        e, nf = r.summary["empty_masks"], r.summary["near_full_masks"]
        L.append(f"- **{r.cfg.name}:** {len(e)} empty mask(s){': ' + ', '.join(e) if e else ''}; "
                 f"{len(nf)} mask(s) covering > {_pct(cfg.full_mask_fraction, 0)}% of the image{': ' + ', '.join(nf) if nf else ''}.")
    L.append("")

    L += ["## 7. Polyp area fraction and number of polyps", "",
          "Area fraction = polyp pixels / all pixels of the binarised mask, in percent.", ""]
    rows = []
    for r in ds.values():
        a = r.summary["area_fraction"]
        rows.append([r.cfg.name, _pct(a["mean"]), _pct(a["median"]), _pct(a["min"]), _pct(a["q25"]), _pct(a["q75"]), _pct(a["max"])])
    L += [md_table(["Dataset", "Mean", "Median", "Min", "Q1", "Q3", "Max"], rows), ""]
    rows = []
    for r in ds.values():
        n, h = r.summary["n_polyps"], r.summary["n_polyps"]["histogram"]
        rows.append([r.cfg.name, _f(n["mean"]), _f(n["median"], 0), int(n["max"]), h.get("1", 0), h.get("2", 0), h.get("3", 0), h.get("4", 0)])
    L += [md_table(["Dataset", "Mean polyps", "Median", "Max", "Masks with 1", "with 2", "with 3", "with 4+"], rows), "",
          "Polyps are counted as 4-connected components (`scipy.ndimage.label`) of the binarised mask. Small isolated specks "
          "left after thresholding noisy masks are counted too, so values above 1 can include annotation noise rather than separate polyps.", ""]

    L += ["## 8. Brightness (for domain-shift analysis)", ""]
    rows = []
    for r in ds.values():
        b, c = r.summary["brightness_gray"], r.summary["channel_means"]
        rows.append([r.cfg.name, _f(b["mean"]), _f(b["std"]), _f(b["min"]), _f(b["max"]), _f(c["r"]["mean"]), _f(c["g"]["mean"]), _f(c["b"]["mean"])])
    L += [md_table(["Dataset", "Grey mean", "Grey std", "Grey min", "Grey max", "R mean", "G mean", "B mean"], rows), ""]
    keys = list(ds)
    if len(keys) == 2:
        a, b = (ds[k].summary["brightness_gray"]["mean"] for k in keys)
        L += [f"On average {ds[keys[0]].cfg.name} images are {abs(a - b):.1f} grey levels {'brighter' if a > b else 'darker'} than {ds[keys[1]].cfg.name} images "
              f"({a:.1f} vs {b:.1f} on a 0-255 scale).", ""]

    L += ["## 9. Duplicates (report only; nothing was deleted)", ""]
    d = result.duplicates
    if d["exact_groups"]:
        L.append(f"Exact duplicates (identical file bytes): {len(d['exact_groups'])} group(s).\n")
        L += [f"- {', '.join(g['files'])}" for g in d["exact_groups"]]
    else:
        L.append("Exact duplicates (identical file bytes): none, within or across the datasets.")
    L += ["", f"Near-duplicates (perceptual hash, Hamming distance <= {cfg.near_duplicate_hamming} of 64 bits). The threshold is deliberately strict; "
          "the table shows how many pairs exist at each distance.", ""]
    near = d["near_pairs"]
    scopes = sorted(d["distance_histogram"])
    rows = []
    for sc in scopes:
        h = d["distance_histogram"][sc]
        rows.append([sc.replace("_", " ")] + [_bucket(h, i) for i in (0, 2, 4, 6)] + [_bucket(h, 8, 1), int((near["scope"] == sc).sum())])
    L += [md_table(["Scope", "Hamming 0-1", "2-3", "4-5", "6-7", "8", f"Near-duplicates (<= {cfg.near_duplicate_hamming})"], rows)
          if rows else "No image pairs within 8 bits.", ""]
    for sc in scopes:
        g = near[near["scope"] == sc]
        if sc == "across_datasets":
            L.append(f"**Across the two datasets:** {len(g)} near-duplicate pair(s).")
            close = d["all_close_pairs"]
            close = close[close["scope"] == sc].head(MAX_LIST)
            if len(close):
                L += ["", "Closest cross-dataset pairs (all listed pairs are within 8 bits; full list in `duplicate_pairs.csv`):", "",
                      md_table(["Image A", "Image B", "Hamming"], [[f"{x.a_dataset}/{x.a_file}", f"{x.b_dataset}/{x.b_file}", x.hamming] for x in close.itertuples()])]
        else:
            seq = ""
            if g["same_sequence"].notna().any():
                seq = f", {int((g['same_sequence'] == True).sum())} within the same video sequence"  # noqa: E712
            L.append(f"**Within {ds[sc[7:]].cfg.name}:** {len(g)} near-duplicate pair(s){seq}.")
            if len(g):
                L += ["", md_table(["Image A", "Image B", "Hamming", "Byte-identical"],
                                   [[x.a_file, x.b_file, x.hamming, x.exact_bytes] for x in g.head(MAX_LIST).itertuples()])]
                if len(g) > MAX_LIST:
                    L.append(f"\n... {len(g) - MAX_LIST} more in `duplicate_pairs.csv`.")
        L.append("")
    for key, lk in result.leakage.items():
        if lk:
            L.append(f"- Leakage check, {ds[key].cfg.name}: {lk['pairs_in_different_splits']} of {lk['near_duplicate_pairs']} within-dataset near-duplicate pairs lie in different splits; "
                     f"{lk['reviewed_as_different_images']} of these were reviewed by hand as different images (pHash false positives), so {lk['leaking_pairs']} remain as possible leakage"
                     + (f" (e.g. {'; '.join(lk['examples'])})." if lk["examples"] else "."))
    L.append("")

    L += ["## 10. Fixed splits", ""]
    rows = []
    for r in ds.values():
        m = r.split_meta
        sz = {k: len(v) for k, v in (r.split or {}).items()}
        rows.append([r.cfg.name, m.get("method", "not created"), m.get("seed", ""),
                     "/".join(str(sz.get(k, "-")) for k in ("train", "val", "test")),
                     m.get("n_sequences", "n/a"), "yes" if m.get("approximate") else "no"])
    L += [md_table(["Dataset", "Method", "Seed", "Train/val/test pairs", "Sequences", "Approximation"], rows), ""]
    for r in ds.values():
        m = r.split_meta
        if r.split_error:
            L.append(f"- **{r.cfg.name}:** split not created. {r.split_error}")
        else:
            L.append(f"- **{r.cfg.name}:** {m['note']}" + (f" Achieved fractions: {m['achieved_fractions']} (target {m['target_fractions']})." if "achieved_fractions" in m else ""))
        if m.get("excluded_unreadable_pairs"):
            L.append(f"  Excluded because unreadable: {', '.join(m['excluded_unreadable_pairs'])}.")
    L += ["", "Each split is checked by assertion: the parts are disjoint, together cover every usable pair exactly once, and (sequence split) no sequence appears in two parts. "
          "Every dataset's complete pair list is also saved as an external-test list (`splits/*_external_test.json`). "
          "Files: `outputs/splits/`.", ""]
    for r in ds.values():
        si = r.sequence_info
        if si.get("n_sequences"):
            fs = si["frames_per_sequence"]
            L.append(f"- Sequence information for {r.cfg.name} ({si['source']}): {si['n_sequences']} sequences, {fs['min']}-{fs['max']} frames each (median {fs['median']:g}).")
        elif r.cfg.sequence_meta is not None:
            L.append(f"- Sequence information for {r.cfg.name} is not available ({si['source']}); the sequence-aware split could not be made.")
    L.append("")

    for r in ds.values():
        x = r.summary.get("crosscheck")
        if x:
            L += [f"## 11. Second image copy cross-check ({r.cfg.name})", "",
                  f"Masks: {x['masks_identical']} of {x['masks_compared']} compared mask pairs are pixel-identical between the main copy and the second copy. "
                  + (f"Images: the second copy of {x['images_second_copy_undecodable']} of {x['images_second_copy_found']} images could not be decoded by Pillow, so images were NOT compared."
                     if x["images_second_copy_undecodable"] else f"Images: {x['images_identical']} of {x['images_compared']} pixel-identical."), ""]

    L += ["## 12. Flagged files", "",
          "Flag meanings: `corrupt_*` cannot be fully decoded; `size_mismatch` image and mask sizes differ; `empty_mask` no polyp pixel after binarising; "
          f"`near_full_mask` more than {_pct(cfg.full_mask_fraction, 0)}% polyp; `not_two_class` binarised mask does not have two classes; `unmatched_*` no partner file; "
          "`exact_duplicate`; `near_duplicate_across_datasets`. Within-dataset near-duplicates and noisy-but-valid masks are not listed per file (see sections 5 and 9, `duplicate_pairs.csv`).", ""]
    fl = result.flagged
    if len(fl):
        L.append(md_table(["Dataset", "File", "Flag", "Detail"], [[x.dataset, x.file, x.flag, x.detail] for x in fl.itertuples()]))
    else:
        L.append("No file was flagged.")
    L.append("")
    return "\n".join(L) + "\n"


def _bucket(hist: dict, start: int, width: int = 2) -> int:
    return int(sum(hist.get(i, 0) for i in range(start, start + width)))
