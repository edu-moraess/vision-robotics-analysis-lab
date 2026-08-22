"""Detector backends: Classical (active) vs ARQTECH (unavailable until checkpoint)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from ..vision.detector import ClassicalDetector, Detection
from .registry import ModelRegistry

class DetectorBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def model_type(self) -> str: ...
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]: ...
    @abstractmethod
    def is_available(self) -> bool: ...

class ClassicalBackend(DetectorBackend):
    def __init__(self, min_area=80, conf_threshold=0.35):
        self._det = ClassicalDetector(min_area=min_area, conf_threshold=conf_threshold)
    @property
    def name(self):
        return "classical-cv-baseline"
    @property
    def model_type(self):
        return "CLASSICAL_BASELINE"
    def is_available(self):
        return True
    def detect(self, frame):
        return self._det.detect(frame)

class ArqtechBackend(DetectorBackend):
    def __init__(self, version="ARQTECH-v0.0-experimental", registry=None):
        self.version = version
        self.registry = registry or ModelRegistry()
        self._record = self.registry.get(version)
    @property
    def name(self):
        return self.version
    @property
    def model_type(self):
        return self._record.get("model_type", "EXPERIMENTAL") if self._record else "EXPERIMENTAL"
    def is_available(self):
        if not self._record or self._record.get("status") != "ACTIVE":
            return False
        return bool(self._record.get("checkpoint_path"))
    def detect(self, frame):
        if not self.is_available():
            raise RuntimeError(f"{self.version} not available for inference — use ClassicalBackend.")
        raise NotImplementedError("ARQTECH forward pass not implemented — no trained weights.")

def select_backend(preference="classical") -> DetectorBackend:
    if preference.lower().startswith("arqtech"):
        b = ArqtechBackend(version=preference if preference != "arqtech" else "ARQTECH-v0.0-experimental")
        if b.is_available():
            return b
        return ClassicalBackend()
    return ClassicalBackend()
