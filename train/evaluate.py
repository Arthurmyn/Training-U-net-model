"""Evaluate a trained run: Dice per image on in-domain val/test and on the other dataset (external)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .config import TrainConfig, load_train_config  # noqa: E402
from .data import Item, SegDataset, limited, load_items  # noqa: E402
from .engine import select_device  # noqa: E402
from .metrics import dice_per_image  # noqa: E402
from .model import UNet  # noqa: E402

log = logging.getLogger("train")


@torch.no_grad()
def predict(model, ds: SegDataset, device, batch_size: int, threshold: float) -> pd.DataFrame:
    """Dice and areas for every image of ``ds`` (in dataset order)."""
    model.eval()
    rows = []
    for img, mask, idx in DataLoader(ds, batch_size=batch_size, shuffle=False):
        logits = model(img.to(device)).float().cpu()
        d = dice_per_image(logits, mask, threshold)
        pred = (torch.sigmoid(logits) > threshold).float()
        for k, i in enumerate(idx.tolist()):
            rows.append({"index": i, "dice": float(d[k]), "gt_area": float(mask[k].mean()), "pred_area": float(pred[k].mean())})
    return pd.DataFrame(rows)


def _qualitative(model, ds: SegDataset, df: pd.DataFrame, device, threshold: float, title: str, path: Path) -> None:
    """Worst / median / best image of an evaluation set: image, ground truth, prediction."""
    order = df.sort_values(["dice", "index"]).reset_index(drop=True)
    picks = [order.iloc[0], order.iloc[len(order) // 2], order.iloc[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5), squeeze=False)
    for r, (label, row) in enumerate(zip(("worst", "median", "best"), picks)):
        img, mask, _ = ds[int(row["index"])]
        with torch.no_grad():
            pred = (torch.sigmoid(model(img.unsqueeze(0).to(device))) > threshold).float().cpu()[0, 0]
        for c, (panel, name) in enumerate(((img.permute(1, 2, 0).numpy(), "image"), (mask[0].numpy(), "ground truth"), (pred.numpy(), "prediction"))):
            ax = axes[r][c]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1) if panel.ndim == 2 else ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
        axes[r][0].set_ylabel(f"{label}\nDice {row['dice']:.2f}", fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def evaluate_run(cfg: TrainConfig, run_dir: Path, limit: Optional[int] = None, progress: bool = True,
                 examples: bool = True) -> Path:
    """Write ``per_image_dice.csv`` and ``eval_summary.json`` into the run folder (best checkpoint by val Dice)."""
    meta = json.loads((run_dir / "run.json").read_text())
    if meta.get("status") != "finished":
        raise RuntimeError(f"{run_dir.name} has not finished training")
    source, seed = meta["source"], int(meta["seed"])
    t = cfg.section("training")
    device = select_device(t.get("device", "auto"))
    thr = float(t.get("threshold", 0.5))
    model = UNet(3, int(cfg.section("model").get("base_channels", 32))).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])

    parts, _ = load_items(cfg, source)
    sets: List[Tuple[str, str, str, List[Item], dict]] = [
        (f"{source}_val", "in_domain", "val", parts["val"], {}),
        (f"{source}_test", "in_domain", "test", parts["test"], {}),
    ]
    for other in cfg.audit.datasets:
        if other == source:
            continue
        o_parts, o_all = load_items(cfg, other)
        own = {i.stem: part for part, items in o_parts.items() for i in items}
        sets.append((f"{other}_external", "external", "all", o_all, own))

    frames, summary = [], {}
    for name, kind, part, items, own_split in sets:
        items = limited(items, limit)
        ds = SegDataset(items, cfg.image_size, cfg.mask_threshold, None, progress)
        df = predict(model, ds, device, int(t["batch_size"]), thr)
        it = [items[i] for i in df["index"]]
        df.insert(0, "run", run_dir.name)
        df.insert(1, "source", source)
        df.insert(2, "seed", seed)
        df.insert(3, "eval_set", name)
        df.insert(4, "kind", kind)
        df.insert(5, "target", name.rsplit("_", 1)[0])
        df["stem"] = [i.stem for i in it]
        df["image_file"] = [i.image_file for i in it]
        df["sequence_id"] = [i.sequence_id for i in it]
        df["target_split"] = [own_split.get(i.stem, part) for i in it]
        frames.append(df.drop(columns=["index"]))
        summary[name] = {"kind": kind, "n_images": int(len(df)), "mean_dice": float(df["dice"].mean()),
                         "median_dice": float(df["dice"].median())}
        log.info("%s  %-16s n=%4d  mean Dice %.4f", run_dir.name, name, len(df), summary[name]["mean_dice"])
        if examples:
            _qualitative(model, ds, df, device, thr, f"{run_dir.name} on {name}", run_dir / f"examples_{name}.png")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(run_dir / "per_image_dice.csv", index=False, float_format="%.6f")
    (run_dir / "eval_summary.json").write_text(json.dumps({"best_epoch": meta["best_epoch"], "threshold": thr, "sets": summary}, indent=2, sort_keys=True))
    return run_dir / "per_image_dice.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a finished run on in-domain and external data.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--run", required=True, help="run folder, e.g. runs/kvasir_seed1")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    evaluate_run(load_train_config(args.config), Path(args.run).resolve(), progress=not args.no_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
