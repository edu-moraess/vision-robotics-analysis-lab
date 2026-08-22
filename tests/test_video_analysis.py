from __future__ import annotations
import sys, tempfile, time
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.camera.video_file import VideoFileSource
from src.vision.video_analysis import VideoAnalyzer

def _make_video(path, n=8, fps=4):
    w, h = 64, 48
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[10:30, 10+i:25+i] = (50, 50, 50)
        writer.write(frame)
    writer.release()

def test_video_seek_and_metadata():
    vp = Path(tempfile.mkdtemp()) / "t.mp4"
    _make_video(vp)
    src = VideoFileSource(str(vp), loop=False)
    assert src.start().online
    assert src.metadata()["filename"] == "t.mp4"
    assert src.seek_frame(0) is not None
    src.pause(); assert src._paused
    src.resume(); src.stop()

def test_video_analyzer_report():
    vp = Path(tempfile.mkdtemp()) / "t.mp4"
    _make_video(vp, n=6)
    src = VideoFileSource(str(vp), loop=False); src.start()
    analyzer = VideoAnalyzer()
    results, fids, tss = [], [], []
    t0 = time.perf_counter()
    for i in range(0, 6, 2):
        pkt = src.seek_frame(i)
        if pkt is None: break
        results.append(analyzer.analyze_frame(pkt.image, run_planner=False))
        fids.append(pkt.frame_id); tss.append(i / 4.0)
    src.stop()
    rep = analyzer.build_report("t.mp4", results, fids, tss, 3, time.perf_counter() - t0)
    assert rep.frames_analyzed >= 1 and isinstance(rep.to_dict(), dict)
