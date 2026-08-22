from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


ARQTECH_LIFECYCLE = (
    "ARCHITECTURE",
    "BOOTSTRAP",
    "REAL DATASET",
    "HUMAN REVIEW",
    "DETECTION TRAINING",
    "VALIDATION",
    "BENCHMARK",
    "PRODUCTION CANDIDATE",
)


@dataclass(frozen=True)
class LifecycleRecord:
    model: str = "ARQTECH"
    version: str = "v0.3-detection-experimental"
    status: str = "ARCHITECTURE"
    dataset_id: Optional[str] = None
    checkpoint: Optional[str] = None
    metrics_status: str = "NOT MEASURED"
    notes: tuple[str, ...] = (
        "v0.3 requires a real human-reviewed detection dataset.",
        "A checkpoint is not evidence of validation or production readiness.",
    )

    def __post_init__(self):
        if self.status not in ARQTECH_LIFECYCLE:
            raise ValueError(f"Unknown lifecycle status: {self.status}")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "version": self.version,
            "status": self.status,
            "dataset_id": self.dataset_id or "NOT AVAILABLE",
            "checkpoint": self.checkpoint or "NOT AVAILABLE",
            "metrics_status": self.metrics_status,
            "notes": list(self.notes),
            "lifecycle": list(ARQTECH_LIFECYCLE),
        }
