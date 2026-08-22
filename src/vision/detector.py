"""Classical Computer Vision detector (baseline). Canny + contours + color. NOT a neural network."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from .geometry import box_iou

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    object_id: Optional[int] = None

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {"class": self.class_name, "confidence": round(self.confidence, 3),
                "bbox": self.bbox, "center": self.center, "width": self.width,
                "height": self.height, "area": self.area, "id": self.object_id}

class ClassicalDetector:
    def __init__(self, min_area: int = 80, conf_threshold: float = 0.35):
        self.min_area = min_area
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
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
            roi = frame[y:y+bh, x:x+bw]
            if roi.size == 0:
                continue
            mean_bgr = tuple(map(int, cv2.mean(roi)[:3]))
            class_name, conf = self._classify(mean_bgr, area, bw, bh)
            if conf < self.conf_threshold:
                continue
            detections.append(Detection(class_name, conf, (x, y, x+bw, y+bh), (x+bw/2.0, y+bh/2.0)))
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
