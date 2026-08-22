"""Tests for camera, occupancy, tracker, preprocessing."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.webcam import WebcamSource
from src.camera.ip_camera import IPCameraSource
from src.vision.preprocessing import Preprocessor
from src.vision.tracker import IoUTracker
from src.vision.detector import Detection
from src.planning.occupancy import build_occupancy_from_mask, build_cost_map
from src.core.pipeline import AnalysisPipeline

def test_webcam_unavailable_device():
    cam = WebcamSource(device_index=99)
    st = cam.start()
    assert st.online is False
    assert cam.read() is None
    cam.stop()

def test_ip_empty_url():
    assert IPCameraSource("").start().online is False

def test_preprocessor_stages():
    prep = Preprocessor(max_side=320).run(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
    assert prep.edges.ndim == 2
    assert "resize" in prep.latency.stages

def test_occupancy_and_cost():
    mask = np.ones((128, 128), dtype=np.uint8)
    mask[40:80, 40:80] = 0
    occ = build_occupancy_from_mask(mask, cell_size=16)
    assert occ.label.startswith("IMAGE-SPACE")
    cost = build_cost_map(occ, inflation=1)
    assert cost.shape == occ.grid.shape

def test_tracker_needs_temporal():
    tr = IoUTracker()
    tr.update([Detection("obstacle", 0.8, (10, 10, 40, 40), (25, 25))])
    tracks2 = tr.update([Detection("obstacle", 0.85, (12, 12, 42, 42), (27, 27))])
    assert any(t.hits >= 2 or t.age > 1 for t in tracks2)

def test_pipeline_latency_breakdown():
    r = AnalysisPipeline().run(np.random.randint(30, 200, (200, 300, 3), dtype=np.uint8))
    assert r.latency.total_ms > 0
    assert "detection" in r.latency.stages
