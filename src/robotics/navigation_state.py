from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class NavigationStatus(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
    REPLANNING = "REPLANNING"
    AVOIDING = "AVOIDING"
    TARGET_REACHED = "TARGET_REACHED"
    STOPPED = "STOPPED"

@dataclass
class NavigationState:
    status: str
    action: str
    path_available: bool
    message: str
    def to_dict(self):
        return {"status": self.status, "action": self.action, "path_available": self.path_available, "message": self.message}

def derive_navigation_state(action: str, path_available: bool, risk_level: str) -> NavigationState:
    action = (action or "STOP").upper()
    if action == "STOP" or risk_level == "CRITICAL":
        return NavigationState(NavigationStatus.STOPPED.value, action, path_available, "Stopped: critical risk or no safe path.")
    if action == "REPLAN":
        return NavigationState(NavigationStatus.REPLANNING.value, action, path_available, "Path blocked or high density — replanning.")
    if not path_available:
        return NavigationState(NavigationStatus.OBSTACLE_DETECTED.value, action, False, "Obstacle representation blocks planned route.")
    if action in ("TURN_LEFT", "TURN_RIGHT"):
        return NavigationState(NavigationStatus.AVOIDING.value, action, path_available, f"Avoiding via {action}.")
    if action == "FORWARD":
        return NavigationState(NavigationStatus.NAVIGATING.value, action, path_available, "Path clear — proceeding.")
    return NavigationState(NavigationStatus.IDLE.value, action, path_available, "Idle.")
