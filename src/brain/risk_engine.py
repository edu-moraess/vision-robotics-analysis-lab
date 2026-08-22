"""Transparent, rule-based risk scoring from monocular image cues."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from ..vision.scene import SceneAnalysis

@dataclass
class RiskAssessment:
    score: float
    level: str
    contributors: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "contributors": {k: round(v, 3) for k, v in self.contributors.items()},
            "notes": self.notes,
        }

class RiskEngine:
    def __init__(self, low=0.30, medium=0.55, high=0.75):
        self.low, self.medium, self.high = low, medium, high

    def assess(self, scene: SceneAnalysis, path_available=True, avg_confidence=None):
        contributors = {}
        notes = [
            "Risk is computed from monocular image cues only.",
            "Not equivalent to real-world collision risk.",
        ]
        c_dens = min(0.40, scene.obstacle_density * 0.55)
        if c_dens > 0.02:
            contributors["Obstacle Density"] = c_dens
        c_free = min(0.30, max(0.0, (0.55 - scene.estimated_free_space_ratio) * 0.70))
        if c_free > 0.02:
            contributors["Low Free Space"] = c_free
        c_dyn = min(0.25, scene.dynamic_object_count * 0.08 + scene.person_count * 0.10)
        if c_dyn > 0.02:
            contributors["Dynamic / Person Presence"] = c_dyn
        if avg_confidence is not None:
            c_unc = min(0.15, max(0.0, (0.55 - avg_confidence) * 0.40))
            if c_unc > 0.02:
                contributors["Detection Uncertainty"] = c_unc
        if not path_available:
            contributors["No Traversable Path"] = 0.20
            notes.append("Planner failed to find a path in image space.")
        score = max(0.0, min(1.0, float(sum(contributors.values()))))
        if score < self.low:
            level = "LOW"
        elif score < self.medium:
            level = "MEDIUM"
        elif score < self.high:
            level = "HIGH"
        else:
            level = "CRITICAL"
        return RiskAssessment(score=score, level=level, contributors=contributors, notes=notes)
