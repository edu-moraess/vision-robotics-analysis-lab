from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FramePacket:
    """Canonical frame contract shared by every input source."""

    image: np.ndarray
    timestamp: float
    frame_id: int
    source: str
    width: int = 0
    height: int = 0
    fps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image is not None and self.image.size > 0:
            self.height, self.width = self.image.shape[:2]
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("resolution", (self.width, self.height))
        self.metadata.setdefault("source", self.source)
        self.metadata.setdefault("frame_id", self.frame_id)
        self.metadata.setdefault("timestamp", self.timestamp)
        if self.fps is not None:
            self.metadata.setdefault("fps", self.fps)

    @property
    def frame(self) -> np.ndarray:
        """Canonical alias requested by the universal input contract."""
        return self.image

    @property
    def resolution(self) -> Tuple[int, int]:
        return int(self.width), int(self.height)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "resolution": self.resolution,
            "fps": self.fps,
            "metadata": dict(self.metadata),
        }


@dataclass
class CameraStatus:
    online: bool
    source: str
    message: str = ""
    resolution: Optional[Tuple[int, int]] = None
    measured_fps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "online": self.online,
            "source": self.source,
            "message": self.message,
            "resolution": self.resolution,
            "measured_fps": self.measured_fps,
            "metadata": dict(self.metadata),
        }


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
