"""Camera source abstraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class FramePacket:
    image: np.ndarray
    timestamp: float
    frame_id: int
    source: str
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if self.image is not None and self.image.size > 0:
            self.height, self.width = self.image.shape[:2]

@dataclass
class CameraStatus:
    online: bool
    source: str
    message: str = ""
    resolution: Optional[Tuple[int, int]] = None
    measured_fps: Optional[float] = None

class CameraSource(ABC):
    @abstractmethod
    def start(self) -> CameraStatus: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def read(self) -> Optional[FramePacket]: ...
    @abstractmethod
    def is_available(self) -> bool: ...
    @abstractmethod
    def status(self) -> CameraStatus: ...
