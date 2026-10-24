"""Check that training will really run on the GPU: ``python -m train.check_gpu [--config ...]``.

Reports the PyTorch build, the device that would be used, runs a few training steps of the configured U-Net with the
configured batch size (and mixed precision on CUDA) to measure speed and peak memory, and estimates the time of one
epoch and of a full run. Exit code 0 for CUDA or MPS, 1 if only the CPU is available (training would be very slow).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict

import torch

from .config import TrainConfig, load_train_config
from .engine import describe_device, make_grad_scaler, select_device
from .metrics import bce_dice_loss
from .model import UNet

log = logging.getLogger("train")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def check(cfg: TrainConfig, steps: int = 6) -> Dict[str, object]:
    """Time ``steps`` training steps (after 2 warm-up steps) at the configured batch size and image size."""
    t = cfg.section("training")
    device = select_device(t.get("device", "auto"))
    batch, size = int(t["batch_size"]), cfg.image_size
    amp = bool(t.get("amp", False)) and device.type == "cuda"
    info: Dict[str, object] = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                               "device": describe_device(device), "device_type": device.type, "batch_size": batch, "image_size": size, "amp": amp}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1024 ** 3, 2)
    model = UNet(3, int(cfg.section("model").get("base_channels", 32))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]))
    scaler = make_grad_scaler(device, amp)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(batch, 3, size, size, generator=gen).to(device)
    y = (torch.rand(batch, 1, size, size, generator=gen) > 0.8).float().to(device)
    model.train()
    seconds = []
    for i in range(steps + 2):
        _sync(device)
        t0 = time.time()
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss = bce_dice_loss(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        _sync(device)
        if i >= 2:
            seconds.append(time.time() - t0)
    step = sum(seconds) / len(seconds)
    info["seconds_per_step"] = round(step, 4)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        info["peak_vram_gb"] = round(peak, 2)
        info["vram_used_share"] = round(peak / info["vram_total_gb"], 2)
    from .data import load_items
    epochs = {}
    for source in cfg.sources:
        n = len(load_items(cfg, source)[0]["train"])
        epochs[source] = {"train_images": n, "steps_per_epoch": -(-n // batch), "seconds_per_epoch_estimate": round(-(-n // batch) * step * 1.15, 1)}
    info["per_source"] = epochs
    max_epochs = int(t.get("max_epochs", 100))
    info["max_epochs"] = max_epochs
    for v in epochs.values():
        v["minutes_for_max_epochs"] = round(v["seconds_per_epoch_estimate"] * max_epochs / 60, 1)
    return info


def report(info: Dict[str, object]) -> int:
    """Print the result; return the process exit code."""
    print(f"PyTorch {info['torch']}  (built for CUDA {info['cuda_build']}), CUDA available: {info['cuda_available']}")
    print(f"Training device: {info['device']}")
    print(f"Test step: batch {info['batch_size']}, {info['image_size']}x{info['image_size']}, mixed precision {'on' if info['amp'] else 'off'} -> {info['seconds_per_step']} s per step")
    if info["device_type"] == "cuda":
        print(f"GPU memory: peak {info['peak_vram_gb']} GB of {info['vram_total_gb']} GB ({int(info['vram_used_share'] * 100)}%)")
        if info["vram_used_share"] > 0.85:
            print("WARNING: memory is nearly full; lower training.batch_size (16 -> 8) in the config to avoid out-of-memory errors.")
    for source, v in info["per_source"].items():
        print(f"  {source}: {v['train_images']} training images, {v['steps_per_epoch']} steps per epoch, about {v['seconds_per_epoch_estimate']} s per epoch, "
              f"up to {v['minutes_for_max_epochs']} min for {info['max_epochs']} epochs")
    if info["device_type"] == "cpu":
        print("\nNO GPU WILL BE USED: training on the CPU is many times slower.")
        if not info["cuda_available"]:
            print("PyTorch does not see a CUDA GPU. Typical causes: the CPU-only build of PyTorch is installed (check `pip list | findstr torch`: a version ending in "
                  "+cpu means CPU-only), or the NVIDIA driver is missing/old (`nvidia-smi` must work). Reinstall PyTorch with a CUDA build, see docs/SETUP_ru.md, section GPU.")
        return 1
    print("\nGPU OK: training will run on the GPU.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check that training runs on the GPU and estimate the time.")
    ap.add_argument("--config", default="train_config.yaml")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="override training.device")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    cfg = load_train_config(args.config)
    if args.device:
        cfg = cfg.override(**{"training.device": args.device})
    return report(check(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
