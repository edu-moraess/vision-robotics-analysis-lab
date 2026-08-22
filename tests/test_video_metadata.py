from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.camera.video_file import VideoFileSource
from src.camera.video_metadata import obtain_video_metadata, empty_metadata

def _make_video(path, n=6, fps=5):
    w, h = 64, 48
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (i * 10, 40, 60)
        writer.write(frame)
    writer.release()

def test_metadata_method_exists():
    assert hasattr(VideoFileSource, "metadata")

def test_valid_mp4_metadata():
    vp = Path(tempfile.mkdtemp()) / "ok.mp4"
    _make_video(vp)
    src = VideoFileSource(str(vp), loop=False)
    assert src.start().online
    m = src.metadata()
    assert m["filename"] == "ok.mp4"
    u = obtain_video_metadata(src)
    assert u["source_class"] == "VideoFileSource"
    assert u["metadata_source"] in ("vs.metadata()", "opencv_capture", "filesystem")
    src.stop()

def test_obtain_without_metadata_method():
    class Stub:
        path = ""
        def is_available(self): return False
    m = obtain_video_metadata(Stub())
    assert m["source_class"] == "Stub"
    assert "filename" in m

def test_empty_file():
    p = Path(tempfile.mkdtemp()) / "empty.mp4"
    p.write_bytes(b"")
    src = VideoFileSource(str(p), loop=False)
    assert not src.start().online
    m = obtain_video_metadata(src, filename="empty.mp4")
    assert m is not None

def test_missing_file():
    src = VideoFileSource("/tmp/does_not_exist_vral_xyz.mp4", loop=False)
    assert not src.start().online
    m = obtain_video_metadata(src)
    assert m["source_class"] == "VideoFileSource"

def test_empty_metadata_helper():
    m = empty_metadata()
    assert m["fps"] is None
    assert m["frame_count"] is None
