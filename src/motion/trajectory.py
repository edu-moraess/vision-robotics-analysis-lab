from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .types import TemporalSample


@dataclass
class Trajectory:
    track_id: int
    class_name: str
    samples: List[TemporalSample] = field(default_factory=list)

    def add(self, sample: TemporalSample, limit: int = 120) -> None:
        self.samples.append(sample)
        if len(self.samples) > limit:
            del self.samples[:-limit]

    @property
    def centers(self) -> List[Tuple[float, float]]:
        return [sample.center for sample in self.samples]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "samples": [sample.to_dict() for sample in self.samples],
            "unit": "IMAGE-SPACE",
        }


class TrajectoryEngine:
    def __init__(self, history_limit: int = 120):
        self.history_limit = max(2, int(history_limit))
        self.trajectories: Dict[int, Trajectory] = {}

    def reset(self) -> None:
        self.trajectories.clear()

    def update(self, tracks: Iterable, timestamp: Optional[float] = None) -> Dict[int, Trajectory]:
        for track in tracks or []:
            track_id = int(track.track_id)
            trajectory = self.trajectories.setdefault(
                track_id, Trajectory(track_id=track_id, class_name=track.class_name),
            )
            trajectory.class_name = track.class_name
            center = tuple(float(v) for v in (track.smoothed_center or track.center))
            velocity = tuple(float(v) for v in (track.velocity or (0.0, 0.0)))
            trajectory.add(TemporalSample(
                timestamp=timestamp,
                frame_id=getattr(track, "last_seen_frame", None),
                center=center,
                velocity=velocity,
                measurement_available=bool(getattr(track, "measurement_available", True)),
            ), limit=self.history_limit)
        return dict(self.trajectories)

    def get(self, track_id: int) -> Optional[Trajectory]:
        return self.trajectories.get(int(track_id))

    def heatmap(self, image_shape: Tuple[int, int], blur_kernel: int = 0) -> dict:
        h, w = int(image_shape[0]), int(image_shape[1])
        heat = np.zeros((max(1, h), max(1, w)), dtype=np.float32)
        samples = 0
        for trajectory in self.trajectories.values():
            centers = trajectory.centers
            for x, y in centers:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w and 0 <= iy < h:
                    heat[iy, ix] += 1.0
                    samples += 1
            for p0, p1 in zip(centers, centers[1:]):
                cv2.line(heat, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, 1)
        if blur_kernel and blur_kernel > 1:
            kernel = int(blur_kernel) | 1
            heat = cv2.GaussianBlur(heat, (kernel, kernel), 0)
        maximum = float(heat.max()) if heat.size else 0.0
        normalized = heat / maximum if maximum > 0 else heat
        return {
            "status": "MEASURED" if samples else "EMPTY",
            "array": normalized,
            "sample_count": samples,
            "track_count": len(self.trajectories),
            "unit": "IMAGE-SPACE PROJECTION",
            "physical_map": False,
            "notes": ["Temporal image-projection heatmap; not a physical circulation map."],
        }

    def summaries(self) -> List[dict]:
        return [trajectory.to_dict() for trajectory in self.trajectories.values()]
