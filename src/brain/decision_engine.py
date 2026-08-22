"""Robot Brain — deterministic decision layer for monocular analysis."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SceneDecision:
    action: str
    confidence: float
    reason: str
    risk: str

    def to_dict(self):
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "risk": self.risk,
        }

class DecisionEngine:
    """Rule-based brain. Actions: FORWARD | TURN_LEFT | TURN_RIGHT | STOP | REPLAN"""

    def decide(self, free_space_ratio, obstacle_density, risk_level, path_available, person_count=0):
        if risk_level == "CRITICAL" or not path_available:
            return SceneDecision("STOP", 0.85, "Critical risk or no traversable path in image space.", risk_level)
        if risk_level == "HIGH" or obstacle_density > 0.55:
            return SceneDecision("REPLAN", 0.70, "High obstacle density; re-evaluate trajectory.", risk_level)
        if person_count > 0 and free_space_ratio < 0.50:
            return SceneDecision("TURN_RIGHT", 0.65, "Person present with limited free space; bias right.", risk_level)
        if free_space_ratio < 0.40:
            return SceneDecision("TURN_LEFT", 0.60, "Low free-space ratio in lower image region.", risk_level)
        return SceneDecision("FORWARD", 0.75, "Adequate free space and acceptable risk level.", risk_level)
