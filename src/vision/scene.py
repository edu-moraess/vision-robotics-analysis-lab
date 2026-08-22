"""Scene understanding — free-space heuristics (image-space, not metric)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from .detector import Detection
from .geometry import lower_roi

@dataclass
class SceneAnalysis:
    image_width: int
    image_height: int
    object_count: int
    obstacle_count: int
    dynamic_object_count: int
    person_count: int
    estimated_free_space_ratio: float
    obstacle_density: float
    dominant_objects: List[str] = field(default_factory=list)
    free_space_mask: Optional[np.ndarray] = None
    processing_notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "image_width": self.image_width, "image_height": self.image_height,
            "object_count": self.object_count, "obstacle_count": self.obstacle_count,
            "dynamic_object_count": self.dynamic_object_count, "person_count": self.person_count,
            "estimated_free_space_ratio": round(self.estimated_free_space_ratio, 4),
            "obstacle_density": round(self.obstacle_density, 4),
            "dominant_objects": self.dominant_objects, "notes": self.processing_notes,
        }

class SceneAnalyzer:
    OBSTACLE_CLASSES = {"obstacle", "wall", "dynamic"}
    DYNAMIC_CLASSES = {"dynamic", "person"}

    def __init__(self, lower_roi_ratio=0.45, edge_threshold=0.12):
        self.lower_roi_ratio = lower_roi_ratio
        self.edge_threshold = edge_threshold

    def analyze(self, image, detections):
        if image is None or image.size == 0:
            raise ValueError("Empty image")
        h, w = image.shape[:2]
        notes = [
            "Free-space uses lower-ROI edge density + detection occupancy.",
            "Values are image-space heuristics, not metric world measurements.",
        ]
        obstacle_count = sum(1 for d in detections if d.class_name in self.OBSTACLE_CLASSES)
        dynamic_count = sum(1 for d in detections if d.class_name in self.DYNAMIC_CLASSES)
        person_count = sum(1 for d in detections if d.class_name == "person")
        class_counts = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        dominant = sorted(class_counts, key=lambda k: class_counts[k], reverse=True)[:3]
        free_mask, free_ratio = self._estimate_free_space(image, detections)
        return SceneAnalysis(w, h, len(detections), obstacle_count, dynamic_count, person_count,
                             float(free_ratio), float(1.0 - free_ratio), dominant, free_mask, notes)

    def _estimate_free_space(self, image, detections):
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)
        roi, y0 = lower_roi(image, self.lower_roi_ratio)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        edges = cv2.Canny(gray, 60, 150)
        edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)
        if edge_density > self.edge_threshold:
            occupied = cv2.dilate((edges > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
            mask[y0:, :] = 1 - occupied
        mid_y = int(h * 0.4)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if y2 < mid_y:
                continue
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 0
        return mask, float(np.count_nonzero(mask)) / max(mask.size, 1)
