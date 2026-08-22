from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


PERCEPTION_CURRENT = "CURRENT"
PERCEPTION_YOLO_BASELINE = "YOLO_BASELINE"
PERCEPTION_FUSION = "FUSION"
PERCEPTION_ARQTECH = "ARQTECH_EXPERIMENTAL"
SMOOTHING_RAW = "RAW"
SMOOTHING_MOVING_AVERAGE = "MOVING_AVERAGE"
SMOOTHING_EXPONENTIAL = "EXPONENTIAL"

# This is intentionally conservative. A COCO class is not a navigation obstacle
# unless it has an explicit relevance configured here.
DEFAULT_CLASS_MAPPING: Dict[str, str] = {
    "person": "HIGH",
    "bicycle": "HIGH",
    "car": "HIGH",
    "motorcycle": "HIGH",
    "bus": "HIGH",
    "truck": "HIGH",
    "train": "MEDIUM",
    "chair": "MEDIUM",
    "couch": "MEDIUM",
    "bed": "MEDIUM",
    "dining table": "MEDIUM",
    "table": "MEDIUM",
    "bench": "MEDIUM",
    "obstacle": "HIGH",
    "wall": "CRITICAL",
    "barrier": "CRITICAL",
    "dynamic": "HIGH",
}


@dataclass
class PerceptionConfig:
    mode: str = PERCEPTION_CURRENT
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "auto"
    image_size: int = 640
    classes: Optional[Sequence[int]] = None
    max_detections: int = 100
    tracker_type: str = "IOU"
    tracking_enabled: bool = False
    tracker_max_age: int = 12
    tracker_min_hits: int = 3
    tracker_iou_threshold: float = 0.30
    smoothing_enabled: bool = False
    smoothing_method: str = SMOOTHING_RAW
    smoothing_window: int = 5
    smoothing_alpha: float = 0.35
    class_mapping: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_MAPPING))
    calibration_status: str = "NOT CALIBRATED"

    def normalized(self) -> "PerceptionConfig":
        mode = str(self.mode or PERCEPTION_CURRENT).upper()
        if mode not in (PERCEPTION_CURRENT, PERCEPTION_YOLO_BASELINE, PERCEPTION_FUSION, PERCEPTION_ARQTECH):
            mode = PERCEPTION_CURRENT
        method = str(self.smoothing_method or SMOOTHING_RAW).upper()
        if method not in (SMOOTHING_RAW, SMOOTHING_MOVING_AVERAGE, SMOOTHING_EXPONENTIAL):
            method = SMOOTHING_RAW
        return PerceptionConfig(
            mode=mode,
            model_path=str(self.model_path or "yolo11n.pt"),
            confidence_threshold=max(0.0, min(1.0, float(self.confidence_threshold))),
            iou_threshold=max(0.0, min(1.0, float(self.iou_threshold))),
            device=str(self.device or "auto"),
            image_size=max(32, int(self.image_size)),
            classes=self.classes,
            max_detections=max(1, int(self.max_detections)),
            tracker_type=str(self.tracker_type or "IOU").upper(),
            tracking_enabled=bool(self.tracking_enabled),
            tracker_max_age=max(1, int(self.tracker_max_age)),
            tracker_min_hits=max(1, int(self.tracker_min_hits)),
            tracker_iou_threshold=max(0.0, min(1.0, float(self.tracker_iou_threshold))),
            smoothing_enabled=bool(self.smoothing_enabled),
            smoothing_method=method,
            smoothing_window=max(1, int(self.smoothing_window)),
            smoothing_alpha=max(0.01, min(1.0, float(self.smoothing_alpha))),
            class_mapping={str(k).lower(): str(v).upper() for k, v in (self.class_mapping or {}).items()},
            calibration_status=str(self.calibration_status or "NOT CALIBRATED"),
        )

    def model_identity(self) -> dict:
        if self.mode == PERCEPTION_ARQTECH:
            return {
                "model": "ARQTECH",
                "model_type": "EXPERIMENTAL PYTORCH MODEL",
                "model_version": "UNAVAILABLE",
                "weights": "NONE",
            }
        if self.mode == PERCEPTION_FUSION:
            return {
                "model": "PERCEPTION FUSION",
                "model_type": "MULTI-SOURCE EVIDENCE",
                "model_version": "RUNTIME",
                "weights": "SOURCE-SPECIFIC",
            }
        if self.mode == PERCEPTION_YOLO_BASELINE:
            return {
                "model": "YOLO",
                "model_type": "EXTERNAL BASELINE",
                "model_version": "UNAVAILABLE",
                "weights": self.model_path,
            }
        return {
            "model": "CURRENT DETECTOR",
            "model_type": "EXISTING CLASSICAL DETECTOR",
            "model_version": "baseline",
            "weights": "NONE",
        }
