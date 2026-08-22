from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from typing import List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    status: str
    method: str
    latency_ms: float
    mask_count: int
    detections: list = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["detections"] = [d.to_dict() if hasattr(d, "to_dict") else d for d in self.detections]
        return payload


class ContourSegmenter:
    """Deterministic bbox-local contour segmentation.

    This is a reproducible image-processing baseline, not a semantic neural
    segmentation model. Its output is marked ESTIMATED and IMAGE-SPACE.
    """

    model_name = "CONTOUR SEGMENTATION"
    model_version = "opencv-contour-v1"
    model_type = "DETERMINISTIC IMAGE PROCESSING BASELINE"

    def __init__(self, min_component_area: int = 12, blur_kernel: int = 3):
        self.min_component_area = max(1, int(min_component_area))
        self.blur_kernel = max(1, int(blur_kernel) | 1)

    @property
    def identity(self) -> dict:
        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "status": "AVAILABLE",
            "semantic": False,
        }

    def segment(self, image: np.ndarray, detections: Sequence, timestamp: Optional[float] = None,
                frame_id: Optional[int] = None) -> SegmentationResult:
        started = time.perf_counter()
        if image is None or image.size == 0:
            return SegmentationResult(
                status="FAILED", method=self.model_name,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                mask_count=0, detections=list(detections or []),
                notes=["Invalid image; original detections preserved."],
            )
        segmented = []
        for detection in detections or []:
            local_mask, area, perimeter = self._segment_bbox(image, detection.bbox)
            segmented.append(replace(
                detection,
                mask=local_mask,
                mask_bbox=tuple(int(v) for v in detection.bbox),
                mask_area_px2=float(area),
                mask_perimeter_px=float(perimeter),
                segmentation_model=self.model_name,
                segmentation_status="ESTIMATED",
            ))
        return SegmentationResult(
            status="ESTIMATED",
            method=self.model_name,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            mask_count=sum(1 for d in segmented if d.mask is not None),
            detections=segmented,
            notes=[
                "Masks are derived from local edges/contours inside detections.",
                "Perimeter is contour-derived in pixels when a contour exists.",
                "This is not semantic segmentation and is not ground truth.",
            ],
        )

    def _segment_bbox(self, image: np.ndarray, bbox) -> tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(bbox[0])))
        y1 = max(0, min(h - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(w, int(bbox[2])))
        y2 = max(y1 + 1, min(h, int(bbox[3])))
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8), 0.0, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_component_area]
        if valid:
            contour = max(valid, key=cv2.contourArea)
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
        else:
            # Preserve a valid, auditable mask even when edges are absent.
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = threshold.astype(np.uint8)
            area = float(np.count_nonzero(mask))
            fallback_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = float(max((cv2.arcLength(c, True) for c in fallback_contours), default=0.0))
        return mask, area, perimeter
