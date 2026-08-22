"""Image-space geometric utilities and GeometryEngine. Units: px / px² unless calibrated."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))

def resize_keep_aspect(image, max_side=1280):
    h, w = image.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale

def lower_roi(image, ratio=0.45):
    h = image.shape[0]
    y0 = int(h * (1.0 - ratio))
    return image[y0:, :], y0

@dataclass
class ObjectGeometry:
    detection_id: Optional[int]
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    width_px: float
    height_px: float
    area_px2: float
    aspect_ratio: float
    perimeter_px: float
    normalized_x: float
    normalized_y: float
    region: str

    def to_dict(self):
        return {
            "id": self.detection_id, "class": self.class_name, "bbox": self.bbox,
            "center": (round(self.center[0], 1), round(self.center[1], 1)),
            "width_px": round(self.width_px, 1), "height_px": round(self.height_px, 1),
            "area_px2": round(self.area_px2, 1), "aspect_ratio": round(self.aspect_ratio, 3),
            "perimeter_px": round(self.perimeter_px, 1),
            "normalized_x": round(self.normalized_x, 3), "normalized_y": round(self.normalized_y, 3),
            "region": self.region, "unit": "pixel",
        }

class GeometryEngine:
    def __init__(self, mode: str = "PIXEL"):
        self.mode = mode

    def analyze(self, detections: Sequence[Any], image_shape: Tuple[int, int]) -> List[ObjectGeometry]:
        h, w = image_shape[:2]
        results = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            bw = float(max(0, x2 - x1)); bh = float(max(0, y2 - y1))
            nx = d.center[0] / max(w, 1); ny = d.center[1] / max(h, 1)
            region = "left" if nx < 0.33 else ("right" if nx > 0.66 else "center")
            results.append(ObjectGeometry(
                getattr(d, "object_id", None), d.class_name, d.bbox, d.center,
                bw, bh, bw * bh, bw / max(bh, 1e-6), 2.0 * (bw + bh), float(nx), float(ny), region,
            ))
        return results
