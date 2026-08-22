"""Reproducible experiment log for ARQTECH research."""
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class ExperimentRecord:
    experiment_id: str
    title: str
    model_version: str
    architecture_notes: str
    dataset_version: Optional[str]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    hardware: Optional[str]
    duration_s: Optional[float]
    status: str
    notes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

class ExperimentLog:
    def __init__(self, root: str = "data/arqtech/experiments"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "experiments.jsonl"

    def create(self, title, model_version="ARQTECH-v0.0-experimental", architecture_notes="",
               dataset_version=None, hyperparameters=None, hardware=None, notes=None):
        rec = ExperimentRecord(
            experiment_id=f"exp_{uuid.uuid4().hex[:8]}", title=title,
            model_version=model_version, architecture_notes=architecture_notes,
            dataset_version=dataset_version, hyperparameters=hyperparameters or {},
            metrics={}, hardware=hardware, duration_s=None, status="PLANNED",
            notes=notes or ["No training executed yet."],
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
        return rec

    def list_experiments(self, limit=50):
        if not self.index_path.exists():
            return []
        rows = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
        return list(reversed(rows[-limit:]))
