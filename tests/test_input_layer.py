from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.input import StreamResolver, InputDescriptor, SourceType, FrameBuffer, mask_url, SmartCapturePolicy, should_capture
from src.input.smart_capture import SmartCaptureState
from src.camera.base import FramePacket

def test_mask_url():
    assert "******" in mask_url("rtsp://user:secret@host/stream")

def test_resolver_webcam():
    r = StreamResolver().resolve(InputDescriptor(SourceType.WEBCAM, "0"))
    assert r.ok and r.openable == "0"

def test_resolver_rejects_youtube_without_ytdlp():
    r = StreamResolver().resolve(InputDescriptor(SourceType.YOUTUBE_LIVE, "https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    if not r.ok:
        assert "not directly supported" in r.message.lower() or "webpage" in r.message.lower() or "yt-dlp" in r.message.lower()

def test_resolver_rtsp_shape():
    assert StreamResolver().resolve(InputDescriptor(SourceType.RTSP, "rtsp://192.168.1.5/stream")).ok

def test_frame_buffer_drops():
    buf = FrameBuffer(capacity=2)
    for i in range(5):
        buf.push(FramePacket(image=np.zeros((4,4,3),dtype=np.uint8), timestamp=0.0, frame_id=i, source="t"))
    assert buf.drops >= 3 and buf.pop_latest() is not None

def test_smart_capture_uncertainty():
    assert should_capture(SmartCapturePolicy(uncertainty_threshold=0.3, cooldown_s=0.0), SmartCaptureState(), [], uncertainty=0.9)
