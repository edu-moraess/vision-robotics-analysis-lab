"""Model registry — honest records and controlled lifecycle transitions."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .status import ModelStatus, can_transition, transition


class ModelRegistry:
    def __init__(self, root: str = "data/models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.jsonl"

    def get(self, version: str):
        for row in self.list_models():
            if version in (row.get("version"), row.get("model_version"), row.get("model_name")):
                return row
        return None

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
            "version": "classical-cv-baseline",
            "status": ModelStatus.ACTIVE.value,
            "lifecycle_status": ModelStatus.ACTIVE.value,
            "model_type": "EXISTING CLASSICAL DETECTOR",
            "notes": ["Production detector in the lab UI."],
        })
        return rows

    def transition(self, version: str, target: str | ModelStatus,
                   notes: Optional[str] = None) -> dict:
        rows = self.list_models()
        target_status = ModelStatus(target).value
        found = False
        updated = None
        for row in rows:
            if version in (row.get("version"), row.get("model_version"), row.get("model_name")):
                current = row.get("lifecycle_status") or row.get("status") or ModelStatus.NOT_TRAINED.value
                row["lifecycle_status"] = transition(current, target_status)
                row["status"] = row["lifecycle_status"]
                row["updated_at"] = time.time()
                if notes:
                    row.setdefault("notes", []).append(notes)
                found = True
                updated = row
                break
        if not found:
            raise KeyError(f"Unknown model version: {version}")
        self._rewrite(rows)
        return dict(updated)

    def register(self, record: dict) -> Path:
        row = dict(record)
        row.setdefault("timestamp", time.time())
        row.setdefault("lifecycle_status", row.get("status", ModelStatus.NOT_TRAINED.value))
        row.setdefault("status", row["lifecycle_status"])
        row.setdefault("version", row.get("model_version") or row.get("model_name"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return self.path

    def _rewrite(self, rows: list[dict]) -> None:
        # Keep the synthetic built-in baseline out of the persistent file.
        persistent = [r for r in rows if r.get("model_name") != "classical-cv-baseline"]
        self.path.write_text(
            "".join(json.dumps(row, default=str) + "\n" for row in persistent),
            encoding="utf-8",
        )


def register_train_result(result: dict, root: str = "data/models") -> Path:
    registry = ModelRegistry(root=root)
    entry = {
        "timestamp": time.time(),
        "experiment_id": result.get("experiment_id"),
        "model_name": result.get("model_name"),
        "model_version": result.get("model_version"),
        "version": result.get("model_version"),
        "status": result.get("status", ModelStatus.TRAINED.value),
        "lifecycle_status": result.get("lifecycle_status", ModelStatus.TRAINED.value),
        "dataset": result.get("dataset"),
        "best_val_acc": result.get("best_val_acc"),
        "best_val_loss": result.get("best_val_loss"),
        "checkpoint_path": result.get("checkpoint_path"),
        "notes": result.get("notes"),
    }
    return registry.register(entry)
