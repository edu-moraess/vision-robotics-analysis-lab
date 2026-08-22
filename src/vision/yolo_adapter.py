from __future__ import annotations

from typing import List, Optional

import numpy as np

from .detector import Detection


class YoloDetector:
    """Thin adapter from Ultralytics results to the normalized Detection contract."""

    model_name = "YOLO"
    model_type = "EXTERNAL BASELINE"

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45, device: str = "auto", image_size: int = 640,
                 classes: Optional[list[int]] = None, max_detections: int = 100):
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = str(device or "auto")
        self.image_size = int(image_size)
        self.classes = classes
        self.max_detections = int(max_detections)
        self._model = None
        self._available = False
        self._error = ""
        self._version = "UNAVAILABLE"
        try:
            import ultralytics
            from ultralytics import YOLO
            self._version = str(getattr(ultralytics, "__version__", "UNKNOWN"))
            self._model = YOLO(self.model_path)
            self._available = True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return bool(self._available and self._model is not None)

    @property
    def error(self) -> str:
        return self._error

    @property
    def identity(self) -> dict:
        return {
            "model": self.model_name,
            "model_type": self.model_type,
            "model_version": self._version,
            "weights": self.model_path,
            "available": self.available,
            "error": self._error or None,
        }

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None,
               frame_id: Optional[int] = None) -> List[Detection]:
        if not self.available:
            raise RuntimeError(self._error or "YOLO baseline unavailable")
        if frame is None or frame.size == 0:
            return []
        kwargs = {
            "source": frame,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "classes": self.classes,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device and self.device.lower() != "auto":
            kwargs["device"] = self.device
        try:
            results = self._model.predict(**kwargs)
        except Exception as exc:
            self._error = f"inference failed: {type(exc).__name__}: {exc}"
            raise
        detections: List[Detection] = []
        for result in results or []:
            names = getattr(result, "names", None) or getattr(self._model, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = self._to_numpy(getattr(boxes, "xyxy", None))
            confs = self._to_numpy(getattr(boxes, "conf", None)).reshape(-1)
            class_ids = self._to_numpy(getattr(boxes, "cls", None)).reshape(-1)
            for idx, coords in enumerate(xyxy):
                if len(coords) < 4:
                    continue
                class_id = int(class_ids[idx]) if idx < len(class_ids) else None
                confidence = float(confs[idx]) if idx < len(confs) else 0.0
                class_name = self._class_name(names, class_id)
                x1, y1, x2, y2 = [int(round(float(v))) for v in coords[:4]]
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        class_id=class_id,
                        source_model=self.model_name,
                        model_version=self._version,
                        model_type=self.model_type,
                        timestamp=timestamp,
                        frame_id=frame_id,
                    )
                )
        return detections[: self.max_detections]

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return np.empty((0,))
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _class_name(names, class_id: Optional[int]) -> str:
        if class_id is None:
            return "unknown"
        if isinstance(names, dict):
            return str(names.get(class_id, names.get(str(class_id), class_id)))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)
