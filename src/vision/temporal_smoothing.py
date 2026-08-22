from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from .perception_config import SMOOTHING_EXPONENTIAL, SMOOTHING_MOVING_AVERAGE, SMOOTHING_RAW


class TemporalSmoother:
    """Smooth track centers without overwriting the raw detector measurements."""

    def __init__(self, enabled: bool = False, method: str = SMOOTHING_RAW,
                 window_size: int = 5, alpha: float = 0.35):
        self.enabled = bool(enabled)
        self.method = str(method or SMOOTHING_RAW).upper()
        self.window_size = max(1, int(window_size))
        self.alpha = max(0.01, min(1.0, float(alpha)))
        self._centers = defaultdict(lambda: deque(maxlen=self.window_size))
        self._last = {}

    def reset(self):
        self._centers.clear()
        self._last.clear()

    def update(self, tracks: Iterable) -> list:
        for track in tracks:
            raw = tuple(float(v) for v in (track.raw_center or track.center))
            track.raw_center = raw
            if not self.enabled or self.method == SMOOTHING_RAW:
                smooth = raw
            elif self.method == SMOOTHING_MOVING_AVERAGE:
                history = self._centers[track.track_id]
                history.append(raw)
                smooth = (
                    sum(p[0] for p in history) / len(history),
                    sum(p[1] for p in history) / len(history),
                )
            elif self.method == SMOOTHING_EXPONENTIAL:
                previous = self._last.get(track.track_id, raw)
                smooth = (
                    self.alpha * raw[0] + (1.0 - self.alpha) * previous[0],
                    self.alpha * raw[1] + (1.0 - self.alpha) * previous[1],
                )
            else:
                smooth = raw
            self._last[track.track_id] = smooth
            track.smoothed_center = smooth
            track.center = smooth
        return list(tracks)

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "window_size": self.window_size,
            "alpha": self.alpha,
        }
