"""Rolling frame cache — temporary, NOT training data."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import time
import numpy as np

@dataclass
class CachedFrame:
    image: np.ndarray
    timestamp: float
    source: str
    frame_id: int

class FrameCache:
    def __init__(self, max_frames: int = 60):
        self.max_frames = max(1, max_frames)
        self._buf: Deque[CachedFrame] = deque(maxlen=self.max_frames)

    def push(self, image: np.ndarray, source: str = "", frame_id: int = 0) -> None:
        if image is None or image.size == 0:
            return
        self._buf.append(CachedFrame(image=image.copy(), timestamp=time.time(), source=source, frame_id=frame_id))

    def latest(self) -> Optional[CachedFrame]:
        return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()
