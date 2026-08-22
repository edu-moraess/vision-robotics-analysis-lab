from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class WorldObject:
    object_id: str
    class_name: str
    confidence: float
    track_id: Optional[int]
    centroid: tuple
    navigation_relevance: str
    position_label: str
    distance: str = "NOT AVAILABLE"
    geometry: dict = field(default_factory=dict)
    motion: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    mask_available: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class WorldModel:
    frame_id: int
    objects: List[WorldObject] = field(default_factory=list)
    free_space_ratio: float = 0.0
    path_available: bool = False
    notes: List[str] = field(default_factory=list)
    robot: dict = field(default_factory=lambda: {"status": "SIMULATION ONLY"})
    target: Optional[dict] = None
    obstacles: List[dict] = field(default_factory=list)
    free_space: dict = field(default_factory=dict)
    trajectories: List[dict] = field(default_factory=list)
    risk_zones: List[dict] = field(default_factory=list)
    occupancy: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "objects": [o.to_dict() for o in self.objects],
            "free_space_ratio": self.free_space_ratio,
            "path_available": self.path_available,
            "notes": self.notes,
            "robot": self.robot,
            "target": self.target,
            "obstacles": self.obstacles,
            "free_space": self.free_space,
            "trajectories": self.trajectories,
            "risk_zones": self.risk_zones,
            "occupancy": self.occupancy,
            "paths": self.paths,
            "space": "IMAGE-SPACE / SIMULATION-SPACE",
            "physical_robot_control": False,
        }

    @staticmethod
    def from_enriched(enriched, free_space_ratio, path_available, frame_id=0,
                      semantic_occupancy=None, trajectories=None, risk_zones=None,
                      occupancy=None, current_path=None, alternative_path=None,
                      simulation=None):
        objs = [WorldObject(
            object_id=str(row.get("track_id") or row.get("id") or i),
            class_name=row.get("class_name", "unknown"),
            confidence=float(row.get("confidence", 0)),
            track_id=row.get("track_id"),
            centroid=(row.get("cx", 0), row.get("cy", 0)),
            navigation_relevance=row.get("navigation_relevance", "UNKNOWN_RELEVANCE"),
            position_label=row.get("position_label", "UNKNOWN"),
            distance=row.get("distance", "NOT AVAILABLE"),
            geometry=row.get("geometry", {}),
            motion=row.get("motion", {}),
            risk=row.get("risk", {}),
            mask_available=bool(row.get("mask_available", False)),
        ) for i, row in enumerate(enriched)]
        return WorldModel(
            frame_id=frame_id, objects=objs,
            free_space_ratio=free_space_ratio, path_available=path_available,
            notes=["Image-space world model. Not metric localization.", "Robot state is simulation-only."],
            robot=simulation or {"status": "SIMULATION ONLY"},
            obstacles=[obj.to_dict() for obj in objs if obj.navigation_relevance not in ("LOW", "UNKNOWN_RELEVANCE")],
            free_space={"ratio": float(free_space_ratio), "unit": "IMAGE-SPACE RATIO"},
            trajectories=list(trajectories or []),
            risk_zones=list(risk_zones or []),
            occupancy=semantic_occupancy or (occupancy.to_dict() if hasattr(occupancy, "to_dict") else (occupancy or {})),
            paths={"current": current_path.to_dict() if hasattr(current_path, "to_dict") else current_path,
                   "alternative": alternative_path.to_dict() if hasattr(alternative_path, "to_dict") else alternative_path},
        )
