from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


FREE = "FREE"
UNKNOWN = "UNKNOWN"
PERSON = "PERSON"
WALL = "WALL"
OBSTACLE = "OBSTACLE"
FURNITURE = "FURNITURE"
VEHICLE = "VEHICLE"


def semantic_label(class_name: str) -> str:
    name = str(class_name or "unknown").lower()
    if name in {"person", "pedestrian", "dynamic"}:
        return PERSON
    if name in {"wall", "barrier"}:
        return WALL
    if name in {"chair", "couch", "bed", "dining table", "table", "bench", "furniture"}:
        return FURNITURE
    if name in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}:
        return VEHICLE
    if name in {"obstacle", "blocked"}:
        return OBSTACLE
    return UNKNOWN


@dataclass
class SemanticOccupancy:
    labels: np.ndarray
    cell_size: int
    image_shape: Tuple[int, int]
    label_counts: Dict[str, int] = field(default_factory=dict)
    source: str = "DETECTION+SEGMENTATION+TRACKING"

    def to_dict(self) -> dict:
        return {
            "shape": tuple(int(v) for v in self.labels.shape),
            "cell_size": self.cell_size,
            "image_shape": self.image_shape,
            "label_counts": dict(self.label_counts),
            "source": self.source,
            "space": "IMAGE-SPACE PROJECTED OCCUPANCY",
            "is_3d": False,
        }

    def label_at(self, row: int, col: int) -> str:
        return str(self.labels[row, col])


def build_semantic_occupancy(image_shape: Tuple[int, int], free_space_mask=None,
                             detections: Iterable = (), tracks: Iterable = (), cell_size: int = 16) -> SemanticOccupancy:
    h, w = int(image_shape[0]), int(image_shape[1])
    cell_size = max(1, int(cell_size))
    gh, gw = max(1, h // cell_size), max(1, w // cell_size)
    labels = np.full((gh, gw), UNKNOWN, dtype="U16")
    if free_space_mask is not None:
        for gy in range(gh):
            for gx in range(gw):
                y0, y1 = gy * cell_size, min(h, (gy + 1) * cell_size)
                x0, x1 = gx * cell_size, min(w, (gx + 1) * cell_size)
                cell = free_space_mask[y0:y1, x0:x1]
                if cell.size and float(np.count_nonzero(cell)) / float(cell.size) >= 0.45:
                    labels[gy, gx] = FREE
    track_by_id = {int(getattr(track, "track_id")): track for track in tracks or []}
    for detection in detections or []:
        label = semantic_label(detection.class_name)
        if label == UNKNOWN:
            continue
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        gx1, gx2 = max(0, x1 // cell_size), min(gw - 1, max(0, (x2 - 1) // cell_size))
        gy1, gy2 = max(0, y1 // cell_size), min(gh - 1, max(0, (y2 - 1) // cell_size))
        mask = getattr(detection, "mask", None)
        mask_bbox = getattr(detection, "mask_bbox", None)
        for gy in range(gy1, gy2 + 1):
            for gx in range(gx1, gx2 + 1):
                should_mark = True
                if mask is not None and mask_bbox is not None:
                    cx = min(w - 1, gx * cell_size + cell_size // 2)
                    cy = min(h - 1, gy * cell_size + cell_size // 2)
                    local_x, local_y = cx - int(mask_bbox[0]), cy - int(mask_bbox[1])
                    should_mark = 0 <= local_y < mask.shape[0] and 0 <= local_x < mask.shape[1] and bool(mask[local_y, local_x])
                if should_mark:
                    labels[gy, gx] = label
    unique, counts = np.unique(labels, return_counts=True)
    return SemanticOccupancy(
        labels=labels, cell_size=cell_size, image_shape=(h, w),
        label_counts={str(k): int(v) for k, v in zip(unique, counts)},
    )
