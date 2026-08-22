from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vision.navigation_relevance import navigation_relevance
from src.vision.scene_narrative import build_narrative, scene_inventory
from src.robotics.navigation_state import derive_navigation_state
from src.core.pipeline import AnalysisPipeline

def test_relevance_not_all_obstacles():
    assert navigation_relevance("person") == "DYNAMIC_AGENT"
    assert navigation_relevance("wall") == "STRUCTURAL"
    assert navigation_relevance("unknown_xyz") == "UNKNOWN_RELEVANCE"

def test_narrative_no_hallucination():
    lines = build_narrative([], 0.8, True, "FORWARD", "LOW")
    assert any("0 object" in l.lower() or "No objects" in l for l in lines)
    assert any("NOT AVAILABLE" in l for l in lines)

def test_inventory():
    class D:
        class_name = "person"
    assert scene_inventory([D(), D()])["person"] == 2

def test_nav_state():
    assert derive_navigation_state("REPLAN", False, "HIGH").status == "REPLANNING"

def test_pipeline_narrative_fields():
    pipe = AnalysisPipeline(min_area=20, conf_threshold=0.2, enable_tracking=False)
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[40:80, 50:100] = (40, 40, 40)
    r = pipe.run(img, run_planner=True)
    assert isinstance(r.narrative, list) and r.navigation_state and r.world_model
