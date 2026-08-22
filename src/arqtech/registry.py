"""Model registry — honest records only."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List

class ModelRegistry:
    def __init__(self, root: str = "data/models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.jsonl"

    def list_models(self):
        rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rows.insert(0, {
            "model_name": "classical-cv-baseline",
            "model_version": "active",
            "status": "ACTIVE",
            "notes": ["Production detector in the lab UI."],
        })
        return rows

def register_train_result(result: dict, root: str = "data/models") -> Path:
    path = Path(root) / "registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "experiment_id": result.get("experiment_id"),
        "model_name": result.get("model_name"),
        "model_version": result.get("model_version"),
        "status": result.get("status"),
        "dataset": result.get("dataset"),
        "best_val_acc": result.get("best_val_acc"),
        "best_val_loss": result.get("best_val_loss"),
        "checkpoint_path": result.get("checkpoint_path"),
        "notes": result.get("notes"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path
