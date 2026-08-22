"""Navigation state machine with transition-based events (debounced)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

IDLE = "IDLE"
NAVIGATING = "NAVIGATING"
OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
PATH_BLOCKED = "PATH_BLOCKED"
REPLANNING = "REPLANNING"
AVOIDING = "AVOIDING"
WAITING = "WAITING"
TARGET_REACHED = "TARGET_REACHED"
STOPPED = "STOPPED"
NO_VALID_PATH = "NO_VALID_PATH"

@dataclass
class PlannerDiagnostics:
    obstacle_density: float = 0.0
    free_space_ratio: float = 0.0
    target_reachable: bool = False
    planner: str = "astar"
    path_length_px: float = 0.0
    nodes_expanded: int = 0
    safety_margin: int = 1
    valid_path: bool = False
    message: str = ""

    def to_dict(self):
        return {
            "obstacle_density": round(self.obstacle_density, 4),
            "free_space_ratio": round(self.free_space_ratio, 4),
            "target_reachable": self.target_reachable,
            "planner": self.planner,
            "path_length_px": round(self.path_length_px, 1),
            "nodes_expanded": self.nodes_expanded,
            "safety_margin": self.safety_margin,
            "valid_path": self.valid_path,
            "message": self.message,
            "space": "IMAGE-SPACE / SIMULATION-SPACE",
        }

@dataclass
class NavigationController:
    state: str = IDLE
    last_event: Optional[str] = None
    events: List[dict] = field(default_factory=list)
    consecutive_blocked: int = 0
    diagnostics: PlannerDiagnostics = field(default_factory=PlannerDiagnostics)

    def reset(self):
        self.state = IDLE
        self.last_event = None
        self.events.clear()
        self.consecutive_blocked = 0
        self.diagnostics = PlannerDiagnostics()

    def update(self, has_path, free_space_ratio, obstacle_density, risk_level="LOW",
               path_length=0.0, nodes=0, frame_id=0):
        diag = PlannerDiagnostics(
            obstacle_density=obstacle_density, free_space_ratio=free_space_ratio,
            target_reachable=has_path, path_length_px=path_length,
            nodes_expanded=nodes, valid_path=has_path, message="",
        )
        prev = self.state
        new_state = prev
        if free_space_ratio < 0.08 or obstacle_density > 0.75:
            new_state = NO_VALID_PATH if not has_path else PATH_BLOCKED
            diag.message = "Saturated occupancy or insufficient free space (image-space)."
        elif not has_path:
            self.consecutive_blocked += 1
            if self.consecutive_blocked == 1:
                new_state = PATH_BLOCKED
                diag.message = "No valid path this frame."
            elif self.consecutive_blocked <= 3:
                new_state = REPLANNING
                diag.message = "Replanning after path failure."
            else:
                new_state = NO_VALID_PATH
                diag.message = "No valid path after replanning attempts."
        else:
            self.consecutive_blocked = 0
            if risk_level in ("HIGH", "CRITICAL"):
                new_state = AVOIDING
                diag.message = "Path exists; elevated risk — cautious navigation."
            else:
                new_state = NAVIGATING
                diag.message = "Valid image-space path."
        if new_state != prev:
            self.events.append({
                "event_type": new_state, "from_state": prev, "to_state": new_state,
                "frame_id": frame_id, "detail": diag.message,
            })
            if len(self.events) > 50:
                self.events = self.events[-50:]
            self.last_event = new_state
        self.state = new_state
        self.diagnostics = diag
        return {
            "status": self.state, "message": diag.message,
            "last_event": self.last_event, "diagnostics": diag.to_dict(),
            "space": "IMAGE-SPACE",
        }

@dataclass
class NavigationStateView:
    status: str
    message: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {"status": self.status, "message": self.message,
                "diagnostics": self.diagnostics, "space": "IMAGE-SPACE"}

def derive_navigation_state(decision_action, path_ok, risk_level):
    action = (decision_action or "").upper()
    if action in ("REPLAN", "REPLANNING"):
        return NavigationStateView(status=REPLANNING, message="Replanning requested (image-space).")
    if action in ("STOP", "STOPPED") or not path_ok:
        if risk_level in ("CRITICAL", "HIGH"):
            status, msg = NO_VALID_PATH, "No valid path or elevated risk (image-space)."
        else:
            status, msg = PATH_BLOCKED, "Path unavailable this frame (image-space)."
    elif action in ("SLOW", "CAUTION", "AVOID"):
        status, msg = AVOIDING, "Cautious navigation due to risk."
    elif path_ok:
        status, msg = NAVIGATING, "Valid image-space path."
    else:
        status, msg = IDLE, "Idle."
    return NavigationStateView(status=status, message=msg)
