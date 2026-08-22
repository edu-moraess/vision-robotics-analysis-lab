from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model_v01 import CLASS_NAMES
from .engine import ARQTECHInference


class ArqtechDetector:
    """Experimental ARQTECH adapter; classification checkpoints are not detectors."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.inference = ARQTECHInference(checkpoint_path=checkpoint_path, device=device)
        self.task = "unknown"
        self._metadata = {}
        if checkpoint_path and self.inference.loaded:
            try:
                import torch
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                self._metadata = dict(payload or {}) if isinstance(payload, dict) else {}
                self.task = str(self._metadata.get("task", "unknown"))
            except Exception:
                self.task = "unknown"

    @property
    def identity(self) -> dict:
        return {
            "model": "ARQTECH",
            "model_type": "EXPERIMENTAL PYTORCH MODEL",
            "model_version": self._metadata.get("version", "v0.2-modular"),
            "weights": self.checkpoint_path or "NONE",
            "task": self.task,
            "available": self.available,
        }

    @property
    def available(self) -> bool:
        return bool(self.inference.loaded and self.task == "object_detection")

    def detect(self, frame: np.ndarray, timestamp=None, frame_id=None) -> List:
        if not self.inference.loaded:
            raise RuntimeError("ARQTECH checkpoint is not loaded")
        if self.task != "object_detection":
            raise RuntimeError(
                f"ARQTECH checkpoint task={self.task} is not an object detector; no detection output claimed"
            )
        raise NotImplementedError(
            "ARQTECH object-detection postprocessing requires reviewed detection annotations"
        )
