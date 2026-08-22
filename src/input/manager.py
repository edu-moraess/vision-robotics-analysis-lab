"""InputManager — resolved sources → CameraSource → FramePacket for perception."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from ..camera.base import CameraSource, FramePacket
from ..camera.webcam import WebcamSource
from ..camera.ip_camera import IPCameraSource
from ..camera.smartphone import SmartphoneCameraSource
from ..camera.video_file import VideoFileSource
from .types import SourceType, InputDescriptor
from .resolver import StreamResolver, ResolveResult
from .frame_buffer import FrameBuffer

@dataclass
class ConnectionDiagnostics:
    connection: str
    decoder: str
    message: str
    resolution: Optional[tuple] = None
    measured_fps: Optional[float] = None
    latency_ms: Optional[float] = None
    source_type: str = ""
    masked_url: str = ""
    openable: Optional[str] = None

    def to_dict(self):
        return {"connection": self.connection, "decoder": self.decoder, "message": self.message,
                "resolution": self.resolution, "measured_fps": self.measured_fps,
                "latency_ms": self.latency_ms, "source_type": self.source_type, "masked_url": self.masked_url}

class InputManager:
    def __init__(self, buffer_capacity: int = 3):
        self.resolver = StreamResolver()
        self.buffer = FrameBuffer(capacity=buffer_capacity)
        self.source = None
        self.descriptor = None
        self.last_diagnostics = ConnectionDiagnostics("IDLE", "N/A", "not connected")

    def test_connection(self, descriptor: InputDescriptor) -> ConnectionDiagnostics:
        resolved = self.resolver.resolve(descriptor)
        if not resolved.ok:
            self.last_diagnostics = ConnectionDiagnostics("FAILED", "N/A", resolved.message,
                source_type=resolved.source_type.value, masked_url=resolved.masked_display)
            return self.last_diagnostics
        cam = self._build(resolved)
        try:
            st = cam.start()
            if not st.online:
                self.last_diagnostics = ConnectionDiagnostics("FAILED", "FAILED", st.message or "cannot open",
                    source_type=resolved.source_type.value, masked_url=resolved.masked_display)
                return self.last_diagnostics
            t0 = time.perf_counter()
            pkt = cam.read()
            dt = (time.perf_counter() - t0) * 1000.0
            cam.stop()
            if pkt is None:
                self.last_diagnostics = ConnectionDiagnostics("FAILED", "FAILED", "no frame decoded",
                    resolution=st.resolution, source_type=resolved.source_type.value, masked_url=resolved.masked_display)
            else:
                self.last_diagnostics = ConnectionDiagnostics("ONLINE", "READY", "probe ok",
                    resolution=(pkt.width, pkt.height), measured_fps=st.measured_fps, latency_ms=round(dt, 2),
                    source_type=resolved.source_type.value, masked_url=resolved.masked_display)
            return self.last_diagnostics
        except Exception as e:
            try: cam.stop()
            except Exception: pass
            self.last_diagnostics = ConnectionDiagnostics("FAILED", "FAILED", str(e),
                source_type=resolved.source_type.value, masked_url=resolved.masked_display)
            return self.last_diagnostics

    def connect(self, descriptor: InputDescriptor) -> ConnectionDiagnostics:
        self.disconnect()
        resolved = self.resolver.resolve(descriptor)
        if not resolved.ok:
            return self.test_connection(descriptor)
        cam = self._build(resolved)
        st = cam.start()
        if not st.online:
            self.last_diagnostics = ConnectionDiagnostics("FAILED", "FAILED", st.message,
                source_type=resolved.source_type.value, masked_url=resolved.masked_display)
            return self.last_diagnostics
        self.source = cam
        self.descriptor = descriptor
        self.last_diagnostics = ConnectionDiagnostics("ONLINE", "READY", "connected",
            resolution=st.resolution, measured_fps=st.measured_fps,
            source_type=resolved.source_type.value, masked_url=resolved.masked_display)
        return self.last_diagnostics

    def disconnect(self):
        if self.source is not None:
            try: self.source.stop()
            except Exception: pass
        self.source = None
        self.last_diagnostics = ConnectionDiagnostics("IDLE", "N/A", "disconnected")

    def read_frame(self):
        if self.source is None: return None
        pkt = self.source.read()
        if pkt is not None:
            self.buffer.push(pkt)
            return self.buffer.pop_latest()
        return None

    def is_online(self):
        return self.source is not None and self.source.is_available()

    def _build(self, resolved: ResolveResult) -> CameraSource:
        t, o = resolved.source_type, resolved.openable or ""
        if t == SourceType.WEBCAM: return WebcamSource(device_index=int(o))
        if t == SourceType.VIDEO_FILE: return VideoFileSource(path=o)
        if t == SourceType.SMARTPHONE: return SmartphoneCameraSource(url=o)
        return IPCameraSource(url=o)
