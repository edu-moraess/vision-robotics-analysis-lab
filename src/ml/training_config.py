from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    experiment_id: str
    model_name: str
    training_mode: str
    dataset_id: str
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    input_resolution: tuple = (640, 640)
    optimizer: str = "adam"
    mixed_precision: bool = False
    early_stopping_patience: int = 10
    notes: List[str] = field(default_factory=list)
    model_version: str = "UNVERSIONED"
    framework: str = "PyTorch"
    dataset_scope: str = "HUMAN_REVIEW_REQUIRED"
    hardware: str = field(default_factory=lambda: platform.machine())
    software: str = field(default_factory=platform.python_version)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_resolution"] = list(self.input_resolution)
        return d


def save_training_config(cfg: TrainingConfig, root: str = "data/experiments") -> Path:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    path = root_p / f"{cfg.experiment_id}_config.json"
    payload = {
        "config": cfg.to_dict(),
        "status": "CONFIGURED_NOT_STARTED",
        "lifecycle_status": "NOT TRAINED",
        "metrics": {},
        "message": "Training is configured but has not executed. No metrics are claimed.",
        "created_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
