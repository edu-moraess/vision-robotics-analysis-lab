"""Video file as CameraSource with seek + metadata."""
from __future__ import annotations
import time
from pathlib import Path as PathLib
from typing import Any, Dict, Optional
import cv2
from .base import CameraSource, CameraStatus, FramePacket

class VideoFileSource(CameraSource):
    def __init__(self, path: str, loop: bool = False):
        self.path = str(PathLib(path).expanduser().resolve()) if path else path
        self.loop = loop
        self._cap = None
        self._frame_id = 0
        self._last_ts = 0.0
        self._fps_ema = None
        self._message = "not started"
        self._file_fps = 0.0
        self._frame_count = 0
        self._width = 0
        self._height = 0
        self._paused = False
        self._last_frame = None

    def start(self) -> CameraStatus:
        self.stop()
        p = PathLib(self.path)
        if not p.exists():
            self._message = f"file not found: {self.path}"
            return self.status()
        if p.stat().st_size == 0:
            self._message = "file is empty (0 bytes)"
            return self.status()
        self._cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._message = (
                f"OpenCV cannot decode this video: {p.name}. "
                "Phone videos (H.264/HEVC) often fail with opencv-headless. "
                "Try re-exporting as MP4 (H.264 + yuv420p) or AVI, or extract frames as images."
            )
            self._cap = None
            return self.status()
        self._file_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._message = (
                f"opened but no frames decoded: {p.name}. "
                "Codec may be unsupported (HEVC/HDR). Re-encode to H.264 MP4."
            )
            self._cap.release()
            self._cap = None
            return self.status()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_id = 0
        self._paused = False
        self._last_frame = frame
        self._message = "online"
        return self.status()

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._message = "stopped"
        self._paused = False

    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def restart(self) -> None:
        if self.is_available():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_id = 0
            self._paused = False
            self._message = "online"

    def seek_frame(self, frame_index: int) -> Optional[FramePacket]:
        if not self.is_available():
            return None
        frame_index = max(0, int(frame_index))
        if self._frame_count > 0:
            frame_index = min(frame_index, max(0, self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            for i in range(frame_index + 1):
                ok, frame = self._cap.read()
                if not ok:
                    self._message = "seek/read failure"
                    return None
            if frame is None:
                return None
        self._frame_id = frame_index + 1
        self._last_frame = frame
        self._message = "online"
        return FramePacket(image=frame, timestamp=time.time(), frame_id=self._frame_id,
                           source=f"video:{PathLib(self.path).name}")

    def seek_ratio(self, ratio: float) -> Optional[FramePacket]:
        ratio = max(0.0, min(1.0, float(ratio)))
        if self._frame_count <= 0:
            return self.read()
        return self.seek_frame(int(ratio * max(0, self._frame_count - 1)))

    def read(self) -> Optional[FramePacket]:
        if not self.is_available():
            return None
        if self._paused:
            if self._last_frame is not None:
                return FramePacket(image=self._last_frame.copy(), timestamp=time.time(),
                                   frame_id=self._frame_id, source=f"video:{PathLib(self.path).name}")
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                self._frame_id = 0
            if not ok or frame is None:
                self._message = "end of video"
                return None
        now = time.time()
        if self._last_ts > 0:
            dt = now - self._last_ts
            if dt > 1e-6:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else 0.8 * self._fps_ema + 0.2 * inst
        self._last_ts = now
        self._frame_id += 1
        self._last_frame = frame
        self._message = "online"
        return FramePacket(image=frame, timestamp=now, frame_id=self._frame_id,
                           source=f"video:{PathLib(self.path).name}")

    def metadata(self) -> Dict[str, Any]:
        p = PathLib(self.path)
        size = p.stat().st_size if p.exists() else None
        duration = None
        if self._file_fps and self._file_fps > 0 and self._frame_count > 0:
            duration = self._frame_count / self._file_fps
        return {
            "filename": p.name if p.exists() else self.path,
            "path": self.path,
            "format": p.suffix.lower().lstrip(".") if p.suffix else "NOT AVAILABLE",
            "duration_s": duration if duration is not None else "NOT AVAILABLE",
            "resolution": (self._width, self._height) if self._width else "NOT AVAILABLE",
            "source_fps": self._file_fps if self._file_fps else "NOT AVAILABLE",
            "frame_count": self._frame_count if self._frame_count else "NOT AVAILABLE",
            "file_size_bytes": size if size is not None else "NOT AVAILABLE",
            "codec": "NOT AVAILABLE",
            "current_frame": self._frame_id,
            "paused": self._paused,
        }

    def status(self) -> CameraStatus:
        res = (self._width, self._height) if self._width else None
        return CameraStatus(online=self.is_available(), source=f"video:{PathLib(self.path).name}",
                            message=self._message, resolution=res, measured_fps=self._fps_ema)
