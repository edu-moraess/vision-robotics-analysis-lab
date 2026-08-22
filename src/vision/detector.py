from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import box_iou


@dataclass
class Detection:
    """Model-agnostic detection contract used by the complete perception stack."""

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    object_id: Optional[int] = None
    class_id: Optional[int] = None
    source_model: str = "CURRENT DETECTOR"
    model_version: str = "baseline"
    model_type: str = "EXISTING CLASSICAL DETECTOR"
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None
    source_models: Tuple[str, ...] = field(default_factory=tuple)
    agreement_count: int = 1
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    mask_bbox: Optional[Tuple[int, int, int, int]] = None
    mask_area_px2: Optional[float] = None
    mask_perimeter_px: Optional[float] = None
    segmentation_model: Optional[str] = None
    segmentation_status: str = "NOT AVAILABLE"

    @property
    def width(self) -> int:
        return max(0, int(self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> int:
        return max(0, int(self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return float(self.center[0])

    @property
    def cy(self) -> float:
        return float(self.center[1])

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 3),
            "bbox": tuple(int(v) for v in self.bbox),
            "center": (round(float(self.center[0]), 3), round(float(self.center[1]), 3)),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "id": self.object_id,
            "track_id": self.object_id,
            "source_model": self.source_model,
            "model": self.source_model,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "source_models": tuple(self.source_models or (self.source_model,)),
            "agreement_count": int(self.agreement_count),
            "mask_available": self.mask is not None,
            "mask_bbox": self.mask_bbox,
            "mask_area_px2": round(float(self.mask_area_px2), 2) if self.mask_area_px2 is not None else None,
            "mask_perimeter_px": round(float(self.mask_perimeter_px), 2) if self.mask_perimeter_px is not None else None,
            "segmentation_model": self.segmentation_model,
            "segmentation_status": self.segmentation_status,
        }


class ClassicalDetector:
    """Existing heuristic detector kept as the always-available fallback."""

    model_name = "CURRENT DETECTOR"
    model_version = "baseline"
    model_type = "EXISTING CLASSICAL DETECTOR"

    def __init__(self, min_area: int = 80, conf_threshold: float = 0.35):
        self.min_area = min_area
        self.conf_threshold = conf_threshold

    @property
    def identity(self) -> dict:
        return {
            "model": self.model_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "weights": "NONE",
            "available": True,
        }

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None,
               frame_id: Optional[int] = None) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 8 or bw > w * 0.85 or bh > h * 0.7:
                continue
            roi = frame[y:y + bh, x:x + bw]
            if roi.size == 0:
                continue
            mean_bgr = tuple(map(int, cv2.mean(roi)[:3]))
            class_name, conf = self._classify(mean_bgr, area, bw, bh)
            if conf < self.conf_threshold:
                continue
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=(x, y, x + bw, y + bh),
                    center=(x + bw / 2.0, y + bh / 2.0),
                    source_model=self.model_name,
                    model_version=self.model_version,
                    model_type=self.model_type,
                    timestamp=timestamp,
                    frame_id=frame_id,
                )
            )
        return self._nms(detections)

    def _classify(self, mean_bgr, area, bw, bh):
        b, g, r = mean_bgr
        if r > 150 and g < 120 and b < 120:
            return "dynamic", min(0.95, 0.55 + area / 5000)
        if r > 140 and 70 < g < 130 and b < 110:
            return "person", min(0.92, 0.50 + area / 4000)
        if abs(r - g) < 25 and abs(g - b) < 25 and r < 100:
            aspect = bw / max(bh, 1)
            if aspect > 1.8 or aspect < 0.4:
                return "wall", 0.70
            return "obstacle", min(0.88, 0.45 + area / 6000)
        return "obstacle", 0.40

    def _nms(self, dets, iou_thresh=0.40):
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if box_iou(best.bbox, d.bbox) < iou_thresh]
        return keep
