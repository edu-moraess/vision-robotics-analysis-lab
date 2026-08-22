from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


STATIC = "STATIC"
MOVING = "MOVING"
APPROACHING = "APPROACHING"
RECEDING = "RECEDING"
CROSSING = "CROSSING"
UNKNOWN = "UNKNOWN"


@dataclass
class TemporalSample:
    timestamp: Optional[float]
    frame_id: Optional[int]
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    measurement_available: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictedPoint:
    horizon_s: float
    center: Tuple[float, float]
    model: str = "CONSTANT VELOCITY"
    unit: str = "IMAGE-SPACE PIXELS"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MotionObservation:
    track_id: int
    class_name: str
    position: Tuple[float, float]
    displacement: Tuple[float, float]
    direction: Tuple[float, float]
    velocity_estimate: Tuple[float, float]
    acceleration_estimate: Tuple[float, float]
    motion_state: str
    unit: str = "IMAGE-SPACE"
    calibrated: bool = False
    history_length: int = 0
    predicted_trajectory: List[PredictedPoint] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["predicted_trajectory"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in self.predicted_trajectory]
        return payload
