from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..vision.scene import SceneAnalysis


SAFE = "SAFE"
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"


@dataclass
class RiskAssessment:
    score: float
    level: str
    contributors: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    object_risks: List[dict] = field(default_factory=list)
    risk_zones: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "contributors": {k: round(v, 3) for k, v in self.contributors.items()},
            "notes": self.notes,
            "object_risks": list(self.object_risks),
            "risk_zones": list(self.risk_zones),
            "space": "IMAGE-SPACE / SIMULATION-SPACE",
            "metric_collision_risk": False,
        }


class RiskEngine:
    def __init__(self, low=0.30, medium=0.55, high=0.75):
        self.low, self.medium, self.high = low, medium, high

    def assess(self, scene: SceneAnalysis, path_available=True, avg_confidence=None,
               detections: Optional[Iterable] = None, motion_observations: Optional[Iterable[dict]] = None,
               image_shape=None, safety_margin: float = 0.10):
        contributors = {}
        notes = [
            "Risk is computed from monocular image cues only.",
            "Not equivalent to real-world collision risk.",
            "Object risk combines class prior, image position, motion and confidence; it is not a physical probability.",
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
        object_risks = self.assess_objects(
            detections=detections or [], motion_observations=motion_observations or [],
            image_shape=image_shape, safety_margin=safety_margin,
        )
        if object_risks:
            contributors["Highest Object Context Risk"] = min(0.30, max(r["score"] for r in object_risks) * 0.30)
        score = max(0.0, min(1.0, float(sum(contributors.values()))))
        level = self.level_for_score(score)
        zones = self.build_risk_zones(image_shape=image_shape, object_risks=object_risks, global_level=level)
        return RiskAssessment(score=score, level=level, contributors=contributors,
                              notes=notes, object_risks=object_risks, risk_zones=zones)

    def assess_objects(self, detections: Iterable, motion_observations: Iterable[dict],
                       image_shape=None, safety_margin: float = 0.10) -> List[dict]:
        h, w = (image_shape[:2] if image_shape is not None else (1, 1))
        by_id = {int(o.get("track_id")): o for o in motion_observations or [] if o.get("track_id") is not None}
        results = []
        for index, detection in enumerate(detections or []):
            x1, y1, x2, y2 = detection.bbox
            cx, cy = detection.center
            normalized_y = float(cy) / max(float(h), 1.0)
            normalized_x = float(cx) / max(float(w), 1.0)
            base_prior = {
                "wall": 0.75, "barrier": 0.75, "obstacle": 0.50,
                "vehicle": 0.45, "car": 0.45, "truck": 0.45,
                "person": 0.35, "dynamic": 0.35,
            }.get(str(detection.class_name).lower(), 0.25)
            proximity = max(0.0, min(1.0, (normalized_y - 0.35) / 0.65))
            confidence_factor = max(0.0, min(1.0, float(detection.confidence)))
            motion = by_id.get(int(detection.object_id)) if detection.object_id is not None else None
            motion_state = str((motion or {}).get("motion_state", "UNKNOWN"))
            speed = math.hypot(*((motion or {}).get("velocity_estimate") or (0.0, 0.0)))
            motion_factor = min(1.0, speed / 80.0)
            if motion_state == "APPROACHING":
                motion_factor = min(1.0, motion_factor + 0.25)
            elif motion_state == "CROSSING":
                motion_factor = min(1.0, motion_factor + 0.15)
            score = max(0.0, min(1.0, 0.45 * base_prior + 0.25 * proximity + 0.15 * confidence_factor + 0.15 * motion_factor + safety_margin * 0.0))
            level = self.level_for_score(score)
            results.append({
                "object_id": detection.object_id if detection.object_id is not None else index,
                "track_id": detection.object_id,
                "class": detection.class_name,
                "score": round(score, 3),
                "level": level,
                "position": (round(normalized_x, 3), round(normalized_y, 3)),
                "bbox": tuple(int(v) for v in detection.bbox),
                "motion_state": motion_state,
                "velocity_unit": "IMAGE-SPACE VELOCITY",
                "source": "DETERMINISTIC CONTEXTUAL RISK",
            })
        return results

    @staticmethod
    def level_for_score(score: float) -> str:
        if score < 0.30:
            return LOW
        if score < 0.55:
            return MEDIUM
        if score < 0.75:
            return HIGH
        return CRITICAL

    @staticmethod
    def build_risk_zones(image_shape=None, object_risks=None, global_level=LOW) -> List[dict]:
        h, w = (image_shape[:2] if image_shape is not None else (1, 1))
        zones = [{
            "zone_id": "GLOBAL",
            "level": global_level,
            "bbox": None,
            "space": "IMAGE-SPACE",
        }]
        for risk in object_risks or []:
            zones.append({
                "zone_id": f"OBJECT_{risk['object_id']}",
                "level": risk["level"],
                "object_id": risk["object_id"],
                "bbox": risk.get("bbox"),
                "space": "IMAGE-SPACE",
            })
        return zones
