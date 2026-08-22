from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SimulationState:
    status: str = "SIMULATION"
    robot_position: Tuple[float, float] = (0.0, 0.0)
    target_position: Tuple[float, float] = (0.0, 0.0)
    heading_rad: float = -math.pi / 2
    navigation_state: str = "IDLE"
    current_path: List[Tuple[int, int]] = field(default_factory=list)
    alternative_path: List[Tuple[int, int]] = field(default_factory=list)
    obstacle_count: int = 0
    risk_zone_count: int = 0
    step_index: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RobotSimulation:
    """Image-space kinematic visualization; never sends physical commands."""

    def __init__(self, step_px: float = 8.0):
        self.step_px = max(0.1, float(step_px))
        self.state = SimulationState(notes=[
            "SIMULATION ONLY.",
            "Robot position and movement are image/simulation-space pixels.",
            "No physical robot control, dynamics or actuator output exists.",
        ])
        self._shape = None

    def reset(self) -> None:
        self.state = SimulationState(notes=[
            "SIMULATION ONLY.",
            "Robot position and movement are image/simulation-space pixels.",
            "No physical robot control, dynamics or actuator output exists.",
        ])
        self._shape = None

    def update(self, image_shape: Tuple[int, int], current_path=None,
               alternative_path=None, navigation_state: Optional[str] = None,
               obstacles: Optional[Iterable] = None, risk_zones: Optional[Iterable] = None) -> SimulationState:
        h, w = int(image_shape[0]), int(image_shape[1])
        self._shape = (h, w)
        if self.state.step_index == 0:
            self.state.robot_position = (float(w // 2), float(max(0, h - 12)))
            self.state.target_position = (float(w // 2), float(min(h - 1, 12)))
        self.state.current_path = list(current_path or [])
        self.state.alternative_path = list(alternative_path or [])
        self.state.navigation_state = str(navigation_state or self.state.navigation_state)
        self.state.obstacle_count = len(list(obstacles or []))
        self.state.risk_zone_count = len(list(risk_zones or []))
        self._step_along_path()
        self.state.step_index += 1
        return self.state

    def render(self, image: np.ndarray, state: Optional[SimulationState] = None) -> np.ndarray:
        state = state or self.state
        canvas = image.copy()
        if state.alternative_path:
            self._draw_polyline(canvas, state.alternative_path, (130, 100, 220), 1)
        if state.current_path:
            self._draw_polyline(canvas, state.current_path, (0, 220, 255), 2)
        rx, ry = [int(round(v)) for v in state.robot_position]
        tx, ty = [int(round(v)) for v in state.target_position]
        cv2.circle(canvas, (rx, ry), 8, (255, 180, 0), -1)
        cv2.circle(canvas, (tx, ty), 7, (0, 220, 0), 2)
        cv2.putText(canvas, "ROBOT SIMULATION", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 210, 80), 2)
        cv2.putText(canvas, f"STATE {state.navigation_state}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 80), 1)
        return canvas

    def _step_along_path(self) -> None:
        if len(self.state.current_path) < 2:
            return
        robot = np.array(self.state.robot_position, dtype=np.float32)
        for waypoint in self.state.current_path:
            target = np.array(waypoint, dtype=np.float32)
            delta = target - robot
            distance = float(np.linalg.norm(delta))
            if distance > 1.0:
                step = min(self.step_px, distance)
                robot += delta / distance * step
                self.state.heading_rad = math.atan2(float(delta[1]), float(delta[0]))
                break
        self.state.robot_position = (float(robot[0]), float(robot[1]))

    @staticmethod
    def _draw_polyline(canvas, points, color, thickness):
        for p0, p1 in zip(points, points[1:]):
            cv2.line(canvas, tuple(map(int, p0)), tuple(map(int, p1)), color, thickness)
