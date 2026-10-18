"""Training loop: seeded, early stopping on validation Dice, resumable, one folder per run."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import SegDataset, limited, load_items
from .metrics import bce_dice_loss, dice_per_image
from .model import UNet, count_parameters

log = logging.getLogger("train")


def select_device(preference: str = "auto") -> torch.device:
    """cuda > mps > cpu when ``auto``; otherwise the requested device (error if unavailable)."""
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available (see docs/SETUP_ru.md)")
    if pref == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(pref)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_name(source: str, seed: int) -> str:
    return f"{source}_seed{seed}"


def environment(device: torch.device) -> dict:
    """Software/hardware record stored with every run."""
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        root = Path(__file__).resolve().parents[1]
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        info["git_commit"], info["git_dirty"] = None, None
    return info


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False, threshold=0.5, progress=False, desc=""):
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates. Returns (loss, mean Dice per image)."""
    train = optimizer is not None
    model.train(train)
    total_loss = total_dice = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for img, mask, _ in tqdm(loader, desc=desc, leave=False, disable=not progress):
            img, mask = img.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(img)
            loss = bce_dice_loss(logits.float(), mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            b = img.shape[0]
            total_loss += loss.item() * b
            total_dice += dice_per_image(logits.float(), mask, threshold).sum().item()
            n += b
    return total_loss / n, total_dice / n


def train_run(cfg: TrainConfig, source: str, seed: int, resume: bool = True, progress: bool = True,
              limit: Optional[int] = None) -> Path:
    """Train one model (``source`` dataset, ``seed``). Returns the run folder.

    Skips runs that already finished. With ``resume`` an interrupted run continues from ``last.pt``.
    The best epoch is chosen by mean Dice per image on the validation split (never the test split).
    """
    t = cfg.section("training")
    run_dir = cfg.runs_dir / run_name(source, seed)
    meta_path = run_dir / "run.json"
    if meta_path.is_file() and json.loads(meta_path.read_text()).get("status") == "finished":
        log.info("%s already finished, skipping", run_dir.name)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for name in ("last.pt", "best.pt", "metrics.csv", "run.json", "per_image_dice.csv", "eval_summary.json"):
            (run_dir / name).unlink(missing_ok=True)

    device = select_device(t.get("device", "auto"))
    set_seed(seed, bool(t.get("deterministic", False)))
    log.info("run %s on %s (torch %s)", run_dir.name, device, torch.__version__)

    parts, _ = load_items(cfg, source)
    size, thr = cfg.image_size, cfg.mask_threshold
    train_ds = SegDataset(limited(parts["train"], limit), size, thr, cfg.section("augmentation"), progress)
    val_ds = SegDataset(limited(parts["val"], limit), size, thr, None, progress)
    gen = torch.Generator().manual_seed(seed)
    nw = int(t.get("num_workers", 0))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(t["batch_size"]), shuffle=True, generator=gen, num_workers=nw,
                              pin_memory=pin, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=int(t["batch_size"]), shuffle=False, num_workers=nw, pin_memory=pin)

    model = UNet(3, int(cfg.section("model").get("base_channels", 32))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=int(t.get("lr_patience", 5)))
    amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if device.type == "cuda" else None

    start_epoch, best, best_epoch, bad = 1, -1.0, 0, 0
    last_path = run_dir / "last.pt"
    if resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best, best_epoch, bad = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["bad"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        log.info("resumed from epoch %d (best val Dice %.4f at epoch %d)", ck["epoch"], best, best_epoch)

    meta = {"status": "running", "source": source, "seed": seed, "limit": limit, "n_train": len(train_ds), "n_val": len(val_ds),
            "n_parameters": count_parameters(model), "image_size": size, "config": {k: cfg.raw.get(k) for k in ("model", "training", "augmentation")},
            "environment": environment(device)}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    metrics_path = run_dir / "metrics.csv"
    t0, reason = time.time(), "max_epochs"
    for epoch in range(start_epoch, int(t["max_epochs"]) + 1):
        e0 = time.time()
        tr_loss, tr_dice = _run_epoch(model, train_loader, device, optimizer, scaler, amp, float(t.get("threshold", 0.5)), progress, f"epoch {epoch} train")
        va_loss, va_dice = _run_epoch(model, val_loader, device, None, None, amp, float(t.get("threshold", 0.5)), progress, f"epoch {epoch} val")
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": tr_loss, "train_dice": tr_dice, "val_loss": va_loss, "val_dice": va_dice,
               "lr": lr, "seconds": round(time.time() - e0, 2)}
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False, float_format="%.6f")
        if va_dice > best + 1e-6:
            best, best_epoch, bad = va_dice, epoch, 0
            _atomic_save({"model": model.state_dict(), "epoch": epoch, "val_dice": va_dice}, run_dir / "best.pt")
        else:
            bad += 1
        scheduler.step(va_dice)
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad": bad, "rng_cpu": torch.get_rng_state()}, last_path)
        log.info("epoch %3d  train loss %.4f  val loss %.4f  val Dice %.4f  (best %.4f @ %d)  %.1fs",
                 epoch, tr_loss, va_loss, va_dice, best, best_epoch, row["seconds"])
        if bad >= int(t["patience"]):
            reason = f"early_stopping (no val improvement for {bad} epochs)"
            break

    meta.update(status="finished", stop_reason=reason, best_epoch=best_epoch, best_val_dice=best,
                epochs_run=int(pd.read_csv(metrics_path)["epoch"].max()), train_seconds_this_session=round(time.time() - t0, 1))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    log.info("finished %s: best val Dice %.4f at epoch %d (%s)", run_dir.name, best, best_epoch, reason)
    return run_dir
