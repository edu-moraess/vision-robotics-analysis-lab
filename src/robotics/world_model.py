"""Image-space world model — not metric SLAM."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
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
    def to_dict(self): return asdict(self)

@dataclass
class WorldModel:
    frame_id: int
    objects: List[WorldObject] = field(default_factory=list)
    free_space_ratio: float = 0.0
    path_available: bool = False
    notes: List[str] = field(default_factory=list)
    def to_dict(self):
        return {"frame_id": self.frame_id, "objects": [o.to_dict() for o in self.objects],
                "free_space_ratio": self.free_space_ratio, "path_available": self.path_available, "notes": self.notes}
    @staticmethod
    def from_enriched(enriched, free_space_ratio, path_available, frame_id=0):
        objs = [WorldObject(
            object_id=str(row.get("track_id") or row.get("id") or i),
            class_name=row.get("class_name", "unknown"),
            confidence=float(row.get("confidence", 0)),
            track_id=row.get("track_id"),
            centroid=(row.get("cx", 0), row.get("cy", 0)),
            navigation_relevance=row.get("navigation_relevance", "UNKNOWN_RELEVANCE"),
            position_label=row.get("position_label", "UNKNOWN"),
            distance=row.get("distance", "NOT AVAILABLE"),
        ) for i, row in enumerate(enriched)]
        return WorldModel(frame_id=frame_id, objects=objs, free_space_ratio=free_space_ratio,
                          path_available=path_available, notes=["Image-space world model. Not metric localization."])
