from __future__ import annotations

from typing import Iterable, List, Tuple

from .types import PredictedPoint


class ConstantVelocityPredictor:
    model_name = "CONSTANT VELOCITY"
    status = "DETERMINISTIC BASELINE"

    def __init__(self, horizons_s: Iterable[float] = (0.25, 0.5, 1.0)):
        self.horizons_s = tuple(float(h) for h in horizons_s if float(h) > 0)

    def predict(self, position: Tuple[float, float], velocity: Tuple[float, float]) -> List[PredictedPoint]:
        px, py = float(position[0]), float(position[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        return [PredictedPoint(
            horizon_s=horizon,
            center=(px + vx * horizon, py + vy * horizon),
            model=self.model_name,
        ) for horizon in self.horizons_s]
