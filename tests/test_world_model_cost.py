import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brain.risk_engine import RiskEngine
from src.planning.cost_map import build_navigation_cost_map
from src.planning.occupancy import build_occupancy_from_mask
from src.planning.semantic_occupancy import build_semantic_occupancy
from src.robotics.world_model import WorldModel
from src.vision.detector import Detection


def test_semantic_occupancy_combines_free_space_and_detection():
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    detection = Detection("person", 0.9, (16, 16, 32, 32), (24, 24))
    semantic = build_semantic_occupancy((64, 64), mask, [detection], [], cell_size=16)
    assert semantic.label_counts["PERSON"] >= 1
    assert semantic.to_dict()["is_3d"] is False


def test_risk_is_contextual_and_cost_map_uses_risk():
    scene = type("Scene", (), {
        "obstacle_density": 0.1,
        "estimated_free_space_ratio": 0.8,
        "dynamic_object_count": 0,
        "person_count": 0,
    })()
    detection = Detection("person", 0.9, (20, 40, 40, 60), (30, 50))
    risk = RiskEngine().assess(scene, detections=[detection], image_shape=(64, 64))
    occupancy = build_occupancy_from_mask(np.ones((64, 64), dtype=np.uint8) * 255, cell_size=16)
    cost = build_navigation_cost_map(occupancy, risk_assessment=risk)
    assert risk.object_risks[0]["level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert cost.to_dict()["metric_cost"] is False
    assert "RISK" in cost.sources


def test_world_model_keeps_simulation_and_metric_limits_explicit():
    wm = WorldModel.from_enriched([], 0.8, True, frame_id=2)
    payload = wm.to_dict()
    assert payload["physical_robot_control"] is False
    assert payload["space"] == "IMAGE-SPACE / SIMULATION-SPACE"
