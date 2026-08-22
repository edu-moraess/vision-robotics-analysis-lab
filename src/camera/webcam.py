"""Local webcam via OpenCV VideoCapture."""
from __future__ import annotations
import time
from typing import Optional
import cv2
from .base import CameraSource, CameraStatus, FramePacket

class WebcamSource(CameraSource):
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.req_width = width
        self.req_height = height
        self._cap = None
        self._frame_id = 0
        self._last_ts = 0.0
        self._fps_ema = None
        self._message = "not started"

    def start(self) -> CameraStatus:
        self.stop()
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            self._message = f"cannot open device {self.device_index}"
            self._cap = None
            return self.status()
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
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
            self._message = "frame drop / read failure"
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
        return FramePacket(
            image=frame, timestamp=now, frame_id=self._frame_id,
            source=f"webcam:{self.device_index}", fps=self._fps_ema,
            metadata={
                "input_type": "WEBCAM",
                "device_index": self.device_index,
                "requested_resolution": (self.req_width, self.req_height),
            },
        )

    def status(self) -> CameraStatus:
        res = None
        if self.is_available() and self._cap is not None:
            res = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return CameraStatus(
            online=self.is_available(), source=f"webcam:{self.device_index}",
            message=self._message, resolution=res, measured_fps=self._fps_ema,
            metadata={"input_type": "WEBCAM", "device_index": self.device_index},
        )
