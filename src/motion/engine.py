from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from .prediction import ConstantVelocityPredictor
from .trajectory import TrajectoryEngine
from .types import (
    APPROACHING,
    CROSSING,
    MOVING,
    RECEDING,
    STATIC,
    UNKNOWN,
    MotionObservation,
)


class MotionEngine:
    """Image-space motion service with deterministic, auditable outputs."""

    def __init__(self, static_speed_threshold: float = 1.0,
                 trajectory_engine: Optional[TrajectoryEngine] = None,
                 predictor: Optional[ConstantVelocityPredictor] = None):
        self.static_speed_threshold = float(static_speed_threshold)
        self.trajectory_engine = trajectory_engine or TrajectoryEngine()
        self.predictor = predictor or ConstantVelocityPredictor()
        self.previous_states: Dict[int, str] = {}
        self.events: List[dict] = []

    def reset(self) -> None:
        self.trajectory_engine.reset()
        self.previous_states.clear()
        self.events.clear()

    def update(self, tracks: Iterable, timestamp: Optional[float] = None) -> tuple[List[MotionObservation], List[dict]]:
        self.events = []
        tracks = list(tracks or [])
        trajectories = self.trajectory_engine.update(tracks, timestamp=timestamp)
        observations: List[MotionObservation] = []
        active_ids = set()
        for track in tracks:
            track_id = int(track.track_id)
            active_ids.add(track_id)
            trajectory = trajectories.get(track_id)
            samples = trajectory.samples if trajectory else []
            position = tuple(float(v) for v in (track.smoothed_center or track.center))
            previous = samples[-2].center if len(samples) >= 2 else position
            dt = self._dt(samples[-2].timestamp, samples[-1].timestamp) if len(samples) >= 2 else 1.0
            displacement = (position[0] - previous[0], position[1] - previous[1])
            velocity = tuple(float(v) for v in (track.velocity or (displacement[0] / dt, displacement[1] / dt)))
            if not getattr(track, "measurement_available", True) and len(samples) >= 2:
                velocity = (displacement[0] / dt, displacement[1] / dt)
            previous_velocity = samples[-2].velocity if len(samples) >= 2 else (0.0, 0.0)
            acceleration = (
                (velocity[0] - previous_velocity[0]) / dt,
                (velocity[1] - previous_velocity[1]) / dt,
            )
            speed = math.hypot(velocity[0], velocity[1])
            direction_norm = math.hypot(displacement[0], displacement[1]) or 1.0
            direction = (displacement[0] / direction_norm, displacement[1] / direction_norm)
            state = self._state(velocity, speed)
            predicted = self.predictor.predict(position, velocity)
            observation = MotionObservation(
                track_id=track_id,
                class_name=track.class_name,
                position=position,
                displacement=displacement,
                direction=direction,
                velocity_estimate=velocity,
                acceleration_estimate=acceleration,
                motion_state=state,
                history_length=len(samples),
                predicted_trajectory=predicted,
                notes=[
                    "IMAGE-SPACE MOTION; no metric calibration applied.",
                    "Prediction uses a deterministic constant-velocity baseline.",
                    "APPROACHING/RECEDING are image-axis proxies until calibration exists.",
                ],
            )
            observations.append(observation)
            self._attach_track_state(track, observation)
            previous_state = self.previous_states.get(track_id)
            if previous_state is not None and previous_state != state:
                self.events.append({
                    "event_type": "MOTION_STATE_CHANGED",
                    "track_id": track_id,
                    "class": track.class_name,
                    "from_state": previous_state,
                    "to_state": state,
                    "timestamp": timestamp,
                    "unit": "IMAGE-SPACE",
                })
            self.previous_states[track_id] = state
        for track_id in set(self.previous_states) - active_ids:
            self.previous_states.pop(track_id, None)
        return observations, list(self.events)

    def heatmap(self, image_shape, blur_kernel: int = 0) -> dict:
        return self.trajectory_engine.heatmap(image_shape, blur_kernel=blur_kernel)

    def trajectories(self) -> List[dict]:
        return self.trajectory_engine.summaries()

    def _state(self, velocity, speed: float) -> str:
        if speed < self.static_speed_threshold:
            return STATIC
        vx, vy = velocity
        if abs(vx) > max(abs(vy) * 1.5, self.static_speed_threshold):
            return CROSSING
        if vy > self.static_speed_threshold:
            return APPROACHING
        if vy < -self.static_speed_threshold:
            return RECEDING
        return MOVING

    @staticmethod
    def _dt(previous: Optional[float], current: Optional[float]) -> float:
        if previous is None or current is None:
            return 1.0
        return max(1e-3, float(current) - float(previous))

    @staticmethod
    def _attach_track_state(track, observation: MotionObservation) -> None:
        track.motion_state = observation.motion_state
        track.motion = observation.to_dict()
        track.predicted_trajectory = [point.to_dict() for point in observation.predicted_trajectory]
        track.trajectory = list(getattr(track, "position_history", []) or [])
