"""Error-analysis material for the main direction (Kvasir-trained model on the 612 CVC-ClinicDB frames).

Selects frames with seed-averaged Dice < threshold (a seeded random sample if there are many) plus extra
reference frames, renders five-panel figures, and writes blank annotation sheets and the category rules.
Annotators must not see Dice values: the figures contain none and the sheets list stems in shuffled order only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .config import TrainConfig  # noqa: E402
from .data import SegDataset  # noqa: E402
from .evaluate import build_eval_sets  # noqa: E402
from .extended import error_dir, ext_section, get_device, load_extended, load_model, predict_masks  # noqa: E402
from audit.load import natural_key  # noqa: E402

CATEGORIES_TEXT = """# Error categories

Assign **one** primary category to every frame (columns `stem`, `category`, `notes` in your sheet). Use the number 1-7.
Look only at the figure (image, ground truth, one prediction per seed); the sheets contain no Dice values on purpose.
Work **independently**: do not discuss with the other annotator until both sheets are finished.

| # | Category | Use it when |
|---|---|---|
| 1 | Missed small polyp | A small ground-truth polyp is present and the model predicts nothing on it (or only something elsewhere). |
| 2 | False detection of folds or reflections | The prediction lies mainly on mucosal folds, ridges or specular highlights, not on the polyp. |
| 3 | Incomplete segmentation | The model finds the polyp but covers only part of it; a clear piece of the polyp is left out. |
| 4 | Inaccurate boundary | The right object is found with a roughly correct shape, but the outline is shifted, too wide or too tight. |
| 5 | Complex background | Debris, bubbles, instruments, blood, bile or strong texture around the polyp seem to mislead the model. |
| 6 | Low contrast | The polyp has almost the same colour or texture as its surroundings, or the frame is dark or blurred. |
| 7 | Several objects in the frame | More than one polyp or polyp-like structure; predictions and ground truth disagree about which one counts. |

Rules for hard cases
- If several categories fit, choose the one that explains the **largest part** of the error and mention the others in `notes`.
- Judge what you see in the predictions of all seeds together; if they differ a lot between seeds, describe the most frequent failure.
- Use exactly one of the seven categories for every frame; if none fits well, choose the closest one and explain in `notes`.
- Disagreements are resolved later in the `resolved_category` column of `disagreements.csv`, together with the other annotator.
"""


def polyp_size_bins(area: pd.Series, q1: float, q3: float) -> pd.Series:
    """small: below Q1 of the ground-truth area of all frames, large: above Q3, medium: in between."""
    return pd.Series(np.where(area < q1, "small", np.where(area > q3, "large", "medium")), index=area.index)


def select_frames(cfg: TrainConfig, df: pd.DataFrame) -> Dict[str, object]:
    """Frame selection (deterministic). Returns the frames table and a selection summary."""
    e = ext_section(cfg)
    src, ext = e.get("error_source", "kvasir"), None
    ext = [k for k in cfg.audit.datasets if k != src][0] + "_external"
    g = df[(df["source"] == src) & (df["eval_set"] == ext)]
    thr, n_sample, n_extra, seed = float(e.get("error_dice_threshold", 0.5)), int(e.get("error_sample_size", 100)), int(e.get("error_extra_frames", 50)), int(e.get("sampling_seed", 42))
    dice = g.pivot_table(index="stem", columns="seed", values="dice")
    mean = dice.mean(axis=1)
    area = g.groupby("stem")["gt_area"].first()
    q1, q3 = float(area.quantile(0.25)), float(area.quantile(0.75))
    low = sorted(mean.index[mean < thr], key=natural_key)
    rest = sorted(mean.index[mean >= thr], key=natural_key)
    rng = np.random.default_rng(seed)
    sampled = sorted(rng.choice(low, n_sample, replace=False).tolist(), key=natural_key) if len(low) > n_sample else list(low)
    extra = sorted(rng.choice(rest, min(n_extra, len(rest)), replace=False).tolist(), key=natural_key)
    rows = []
    for group, stems in (("low_dice", sampled), ("reference", extra)):
        for s in stems:
            sub = g[g["stem"] == s].set_index("seed")
            row = {"stem": s, "sequence_id": g.loc[g["stem"] == s, "sequence_id"].iloc[0], "group": group,
                   "polyp_size_bin": polyp_size_bins(pd.Series([area[s]]), q1, q3).iloc[0]}
            row.update({f"dice_seed{k}": float(sub.loc[k, "dice"]) for k in sorted(sub.index)})
            row["mean_dice"] = float(mean[s])
            row.update({f"gt_area_seed{k}": float(sub.loc[k, "gt_area"]) for k in sorted(sub.index)})
            row.update({f"pred_area_seed{k}": float(sub.loc[k, "pred_area"]) for k in sorted(sub.index)})
            rows.append(row)
    info = {"model": f"{src}-trained", "evaluation_set": ext, "total_frames": int(len(mean)), "dice_threshold": thr,
            "frames_below_threshold": int(len(low)), "sampled_low_dice_frames": len(sampled), "reference_frames": len(extra),
            "reference_frames_drawn_from": f"frames with seed-averaged Dice >= {thr} ({len(rest)} available)", "sampling_seed": seed,
            "size_bins": f"small < {q1:.4f} (Q1 of GT area fraction at 256 px), large > {q3:.4f} (Q3), medium otherwise"}
    return {"frames": pd.DataFrame(rows), "info": info, "src": src, "ext": ext}


def render_panels(cfg: TrainConfig, frames: pd.DataFrame, src: str, ext: str, out: Path) -> None:
    """One PNG per frame: image, GT overlay, prediction overlay for each seed. No Dice values are drawn."""
    e = ext_section(cfg)
    device = get_device(cfg)
    items = next(items for name, _, _, items, _ in build_eval_sets(cfg, src) if name == ext)
    index = {it.stem: i for i, it in enumerate(items)}
    chosen = [items[index[s]] for s in frames["stem"]]
    ds = SegDataset(chosen, cfg.image_size, cfg.mask_threshold, None, False)
    seeds = sorted(int(c.replace("dice_seed", "")) for c in frames.columns if c.startswith("dice_seed"))
    preds: Dict[int, np.ndarray] = {}
    for seed in seeds:
        model = load_model(cfg.runs_dir / f"{src}_seed{seed}", int(cfg.section("model").get("base_channels", 32)), device)
        preds[seed], _ = predict_masks(model, ds, device, int(e.get("batch_size", 16)), float(cfg.section("training").get("threshold", 0.5)))
    out.mkdir(parents=True, exist_ok=True)
    green, orange = np.array([0.10, 0.65, 0.30]), np.array([0.92, 0.41, 0.20])

    def tint(img: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
        o = img.copy()
        o[mask] = 0.55 * o[mask] + 0.45 * color
        return o

    for i, stem in enumerate(frames["stem"]):
        img, gt = ds.images[i] / 255.0, ds.masks[i].astype(bool)
        fig, axes = plt.subplots(1, 2 + len(seeds), figsize=(3.2 * (2 + len(seeds)), 3.5), facecolor="white")
        panels = [("Image", img), ("Ground truth (green)", tint(img, gt, green))] + \
                 [(f"Prediction, seed {s} (orange)", tint(img, preds[s][i], orange)) for s in seeds]
        for k, (ax, (title, panel)) in enumerate(zip(axes, panels)):
            ax.imshow(panel)
            if k >= 2:
                ax.contour(gt.astype(float), levels=[0.5], colors=[green], linewidths=1.0)      # ground-truth outline for comparison
            ax.set_title(title, fontsize=9, color="black")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Frame {stem}", fontsize=10, color="black", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=110, facecolor="white", metadata={"Software": None})
        plt.close(fig)


def write_sheets(frames: pd.DataFrame, directory: Path, seed: int) -> None:
    """Blank annotation sheets: stems in shuffled order (different for A and B), empty category and notes."""
    for name, offset in (("annotator_A.csv", 0), ("annotator_B.csv", 1)):
        stems = np.random.default_rng(seed + offset).permutation(frames["stem"].to_numpy())
        pd.DataFrame({"stem": stems, "category": "", "notes": ""}).to_csv(directory / name, index=False)


def run(cfg: TrainConfig) -> List[str]:
    """Write frames.csv, selection.json, frames/<stem>.png, annotator_A.csv, annotator_B.csv and CATEGORIES.md."""
    directory = error_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    sel = select_frames(cfg, load_extended(cfg))
    frames, info = sel["frames"], sel["info"]
    frames.to_csv(directory / "frames.csv", index=False, float_format="%.6f")
    (directory / "selection.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_panels(cfg, frames, sel["src"], sel["ext"], directory / "frames")
    write_sheets(frames, directory, int(ext_section(cfg).get("sampling_seed", 42)))
    (directory / "CATEGORIES.md").write_text(CATEGORIES_TEXT, encoding="utf-8")
    return ["frames.csv", "selection.json", "frames/", "annotator_A.csv", "annotator_B.csv", "CATEGORIES.md"]
