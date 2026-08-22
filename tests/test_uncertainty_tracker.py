"""Uncertainty and tracker tests."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brain.uncertainty import UncertaintyEngine
from src.vision.detector import Detection
from src.vision.tracker import IoUTracker
from src.planning.occupancy import build_occupancy_from_mask, build_cost_map
from src.core.pipeline import AnalysisPipeline

def test_uncertainty_empty_detections():
    u = UncertaintyEngine().assess([], 0.5, 0.3, True, 0, 0.7)
    assert 0.0 <= u.overall <= 1.0
    assert "detection" in u.contributors

def test_uncertainty_with_detections():
    dets = [Detection("obstacle", 0.9, (10, 10, 40, 40), (25, 25))]
    u = UncertaintyEngine().assess(dets, 0.7, 0.2, True, 50, 0.75)
    assert u.detection_uncertainty < 0.5

def test_tracker_single_frame():
    tr = IoUTracker(min_hits=2)
    out = tr.update([Detection("obstacle", 0.8, (0, 0, 20, 20), (10, 10))])
    assert isinstance(out, list)

def test_tracker_two_frames():
    tr = IoUTracker(min_hits=1, iou_threshold=0.2)
    tr.update([Detection("obstacle", 0.8, (0, 0, 20, 20), (10, 10))])
    out = tr.update([Detection("obstacle", 0.85, (2, 2, 22, 22), (12, 12))])
    assert isinstance(out, list)

def test_occupancy_and_cost():
    mask = np.ones((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 0
    grid = build_occupancy_from_mask(mask, cell_size=8)
    cost = build_cost_map(grid, inflation=1)
    assert cost.shape == grid.grid.shape
    assert grid.free_ratio() > 0

def test_pipeline_uncertainty_field():
    img = np.random.randint(20, 200, (96, 128, 3), dtype=np.uint8)
    r = AnalysisPipeline(cell_size=16).run(img)
    assert r.uncertainty is not None
    assert "uncertainty_overall" in r.metrics()
