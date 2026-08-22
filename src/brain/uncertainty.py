"""
Uncertainty as a first-class engineering quantity.

Confidence scores from the classical detector are heuristic ranking
signals, NOT calibrated probabilities. This module aggregates them
into an explicit uncertainty report without inventing statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..vision.detector import Detection


@dataclass
class UncertaintyReport:
    """Structured uncertainty — not a calibrated probability model."""

    detection_uncertainty: float
    scene_uncertainty: float
    planner_uncertainty: float
    decision_uncertainty: float
    overall: float
    contributors: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detection_uncertainty": round(self.detection_uncertainty, 3),
            "scene_uncertainty": round(self.scene_uncertainty, 3),
            "planner_uncertainty": round(self.planner_uncertainty, 3),
            "decision_uncertainty": round(self.decision_uncertainty, 3),
            "overall": round(self.overall, 3),
            "contributors": {k: round(v, 3) for k, v in self.contributors.items()},
            "notes": self.notes,
        }


class UncertaintyEngine:
    """Deterministic aggregation of heuristic uncertainty signals."""

    def assess(
        self,
        detections: List[Detection],
        free_space_ratio: float,
        obstacle_density: float,
        path_success: bool,
        path_nodes: int = 0,
        decision_confidence: float = 0.5,
    ) -> UncertaintyReport:
        notes = [
            "Detector confidence is a classical-CV ranking score, not a calibrated probability.",
            "Uncertainty values are engineering heuristics for observability.",
        ]
        contributors: Dict[str, float] = {}

        if detections:
            confs = np.array([d.confidence for d in detections], dtype=np.float64)
            det_u = float(np.clip(1.0 - confs.mean(), 0.0, 1.0))
            if len(confs) > 1:
                det_u = float(np.clip(det_u + 0.25 * confs.std(), 0.0, 1.0))
        else:
            det_u = 0.55
            notes.append("No detections — elevated detection uncertainty.")

        contributors["detection"] = det_u

        scene_u = float(
            np.clip(abs(free_space_ratio - 0.5) * 0.6 + obstacle_density * 0.3, 0.0, 1.0)
        )
        contributors["scene"] = scene_u

        if not path_success:
            plan_u = 0.85
            notes.append("Planner failed — high planner uncertainty.")
        elif path_nodes > 500:
            plan_u = 0.45
        else:
            plan_u = 0.15
        contributors["planner"] = plan_u

        dec_u = float(np.clip(1.0 - decision_confidence, 0.0, 1.0))
        contributors["decision"] = dec_u

        overall = float(
            np.clip(
                0.35 * det_u + 0.25 * scene_u + 0.25 * plan_u + 0.15 * dec_u,
                0.0,
                1.0,
            )
        )

        return UncertaintyReport(
            detection_uncertainty=det_u,
            scene_uncertainty=scene_u,
            planner_uncertainty=plan_u,
            decision_uncertainty=dec_u,
            overall=overall,
            contributors=contributors,
            notes=notes,
        )
