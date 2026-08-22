"""Integration and unit tests."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.detector import ClassicalDetector
from src.vision.geometry import box_iou
from src.vision.scene import SceneAnalyzer
from src.brain.risk_engine import RiskEngine
from src.brain.decision_engine import DecisionEngine
from src.planning.image_planner import ImageSpacePlanner
from src.core.pipeline import AnalysisPipeline

def test_box_iou():
    assert 0.1 < box_iou((0, 0, 10, 10), (5, 5, 15, 15)) < 0.3

def test_detector_empty():
    assert ClassicalDetector().detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []

def test_detector_noise():
    assert isinstance(ClassicalDetector(min_area=50).detect(
        np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)), list)

def test_scene_analyzer():
    scene = SceneAnalyzer().analyze(np.random.randint(40, 180, (240, 320, 3), dtype=np.uint8), [])
    assert 0.0 <= scene.estimated_free_space_ratio <= 1.0

def test_risk_engine():
    from src.vision.scene import SceneAnalysis
    r = RiskEngine().assess(SceneAnalysis(320, 240, 0, 0, 0, 0, 0.8, 0.2), True, 0.7)
    assert r.level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

def test_decision_engine():
    assert DecisionEngine().decide(0.7, 0.2, "LOW", True, 0).action == "FORWARD"

def test_planner():
    r = ImageSpacePlanner(cell_size=16).plan(np.ones((160, 240), dtype=np.uint8), "astar")
    assert r.success and len(r.path_px) > 0

def test_pipeline():
    result = AnalysisPipeline().run(np.random.randint(30, 200, (240, 320, 3), dtype=np.uint8))
    assert result.processing_time_ms > 0
    assert result.decision.action in ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "REPLAN")
