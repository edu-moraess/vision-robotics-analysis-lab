from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vision.detector import Detection
from src.vision.nms import nms_detections
from src.vision.tracker import IoUTracker, STATUS_CONFIRMED
from src.planning.obstacle_fusion import fuse_obstacles, fusion_stats
from src.robotics.navigation_state import NavigationController, PATH_BLOCKED, NO_VALID_PATH
from src.core.pipeline import AnalysisPipeline

def test_nms_reduces_overlaps():
    dets = [
        Detection("obstacle", 0.9, (10, 10, 50, 50), (30, 30)),
        Detection("obstacle", 0.8, (12, 12, 52, 52), (32, 32)),
        Detection("person", 0.7, (100, 100, 140, 180), (120, 140)),
    ]
    out = nms_detections(dets, iou_threshold=0.4)
    assert len(out) <= 2

def test_tracker_confirm_not_left_immediately():
    tr = IoUTracker(max_age=5, min_hits=2, temp_lost_age=2)
    d = Detection("person", 0.6, (20, 20, 60, 80), (40, 50))
    tr.update([d]); tr.update([d])
    tr.update([]); tr.update([])
    assert not any(e["event_type"] == "OBJECT_LEFT" for e in tr.events)
    left_seen = False
    for _ in range(15):
        tr.update([])
        if any(e["event_type"] == "OBJECT_LEFT" for e in tr.events):
            left_seen = True; break
    assert left_seen

def test_obstacle_fusion_merges():
    dets = [
        Detection("obstacle", 0.8, (10, 10, 40, 40), (25, 25)),
        Detection("obstacle", 0.7, (12, 12, 42, 42), (27, 27)),
        Detection("wall", 0.6, (200, 200, 250, 280), (225, 240)),
    ]
    fused = fuse_obstacles(dets, min_relevance="MEDIUM")
    assert len(fused) <= 2
    assert fusion_stats(3, fused)["raw_detections"] == 3

def test_nav_controller_debounces():
    ctrl = NavigationController()
    ctrl.update(False, 0.1, 0.8, "HIGH")
    ctrl.update(False, 0.1, 0.8, "HIGH")
    assert len(ctrl.events) >= 1

def test_pipeline_still_runs():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[40:80, 40:80] = 180
    r = AnalysisPipeline(enable_tracking=True, min_area=50, conf_threshold=0.35).run(img)
    assert r.processing_time_ms > 0
    assert isinstance(r.fused_obstacles, list)
    assert "status" in (r.navigation_state or {})
