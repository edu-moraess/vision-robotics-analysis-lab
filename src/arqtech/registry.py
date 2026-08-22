"""ARQTECH Model Registry — no ACTIVE neural model without real checkpoint."""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

class ModelStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"
    SCAFFOLD = "SCAFFOLD"

@dataclass
class ModelRecord:
    version: str
    status: str
    model_type: str
    dataset_version: Optional[str] = None
    checkpoint_path: Optional[str] = None
    architecture: Optional[str] = None
    parameter_count: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    hardware: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

class ModelRegistry:
    def __init__(self, root: str = "data/arqtech/registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "models.jsonl"
        self._ensure_baseline()

    def _ensure_baseline(self) -> None:
        if self.index_path.exists() and self.index_path.stat().st_size > 0:
            return
        self.register(ModelRecord(
            version="classical-cv-baseline", status=ModelStatus.ACTIVE.value,
            model_type="CLASSICAL_BASELINE",
            architecture="Canny + contours + color heuristics + NMS",
            parameter_count=0, metrics={},
            configuration={"min_area": 80, "conf_threshold": 0.35},
            notes=["Active production detector. Not a neural network."],
        ))
        self.register(ModelRecord(
            version="ARQTECH-v0.0-scaffold", status=ModelStatus.SCAFFOLD.value,
            model_type="SCAFFOLD",
            architecture="Modular design only — not trained",
            notes=["ARQTECH has NOT been trained. No checkpoint. No mAP."],
        ))

    def register(self, record: ModelRecord) -> None:
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def list_models(self) -> List[dict]:
        if not self.index_path.exists():
            return []
        out = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
        return out

    def get(self, version: str):
        for m in self.list_models():
            if m.get("version") == version:
                return m
        return None

    def active_models(self):
        return [m for m in self.list_models() if m.get("status") == ModelStatus.ACTIVE.value]
