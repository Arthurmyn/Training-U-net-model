"""Training configuration (train_config.yaml) on top of the audit configuration (config.yaml)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from audit.config import AuditConfig, load_config


@dataclass
class TrainConfig:
    """All settings of the training experiment."""

    raw: Dict[str, Any]
    audit: AuditConfig
    base_dir: Path

    @property
    def runs_dir(self) -> Path:
        return (self.base_dir / self.raw.get("runs_dir", "runs")).resolve()

    @property
    def results_dir(self) -> Path:
        return (self.base_dir / self.raw.get("results_dir", "results")).resolve()

    @property
    def splits_dir(self) -> Path:
        return self.audit.output_dir / "splits"

    @property
    def sources(self) -> List[str]:
        return list(self.raw["sources"])

    @property
    def seeds(self) -> List[int]:
        return [int(s) for s in self.raw["seeds"]]

    @property
    def image_size(self) -> int:
        return self.audit.preprocess_size

    @property
    def mask_threshold(self) -> int:
        return self.audit.mask_binarize

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {})

    def override(self, **changes: Any) -> "TrainConfig":
        """Copy with dotted-key overrides, e.g. ``override(**{"training.max_epochs": 2, "runs_dir": "runs_smoke"})``."""
        raw = copy.deepcopy(self.raw)
        for key, value in changes.items():
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return TrainConfig(raw=raw, audit=self.audit, base_dir=self.base_dir)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge: dictionaries are merged key by key, every other value (lists too) is replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load_train_config(path: str | Path = "train_config.yaml") -> TrainConfig:
    """Load a training config and the audit config it points to.

    Optional keys of a config file: ``project_root`` (folder that paths are relative to; default: the file's folder,
    use ``..`` for a file in ``configs/``) and ``base`` (another config, relative to the project root, whose settings
    are inherited; only the differences need to be written). ``audit_config`` is relative to the project root.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_dir = (path.parent / raw["project_root"]).resolve() if raw.get("project_root") else path.parent
    if raw.get("base"):
        base_raw = yaml.safe_load((base_dir / raw["base"]).read_text(encoding="utf-8"))
        raw = _merge(base_raw, {k: v for k, v in raw.items() if k not in ("base", "project_root")})
    audit = load_config(base_dir / raw.get("audit_config", "config.yaml"))
    return TrainConfig(raw=raw, audit=audit, base_dir=base_dir)
