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


def load_train_config(path: str | Path = "train_config.yaml") -> TrainConfig:
    """Load train_config.yaml and the audit config it points to (paths relative to the file)."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    audit = load_config(path.parent / raw.get("audit_config", "config.yaml"))
    return TrainConfig(raw=raw, audit=audit, base_dir=path.parent)
