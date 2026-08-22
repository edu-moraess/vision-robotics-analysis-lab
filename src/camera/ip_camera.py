"""IP / network camera via OpenCV."""
from __future__ import annotations
import time
from typing import Optional
import cv2
from .base import CameraSource, CameraStatus, FramePacket

class IPCameraSource(CameraSource):
    def __init__(self, url: str):
        self.url = url.strip()
        self._cap = None
        self._frame_id = 0
        self._last_ts = 0.0
        self._fps_ema = None
        self._message = "not started"

    def start(self) -> CameraStatus:
        self.stop()
        if not self.url:
            self._message = "empty URL"
            return self.status()
        self._cap = cv2.VideoCapture(self.url)
        if not self._cap.isOpened():
            self._message = f"cannot open stream: {self.url[:48]}"
            self._cap = None
            return self.status()
        self._message = "online"
        return self.status()

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._message = "stopped"

    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        if not self.is_available():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._message = "stream interrupted / drop"
            return None
        now = time.time()
        if self._last_ts > 0:
            dt = now - self._last_ts
            if dt > 1e-6:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else 0.8 * self._fps_ema + 0.2 * inst
        self._last_ts = now
        self._frame_id += 1
        self._message = "online"
        return FramePacket(image=frame, timestamp=now, frame_id=self._frame_id, source=f"ip:{self.url[:32]}")

    def status(self) -> CameraStatus:
        res = None
        if self.is_available() and self._cap is not None:
            res = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return CameraStatus(online=self.is_available(), source=f"ip:{self.url[:40]}", message=self._message, resolution=res, measured_fps=self._fps_ema)
