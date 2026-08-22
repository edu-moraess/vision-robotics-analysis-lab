from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.learning import ExperienceMemory
from src.ml.dataset_builder import DatasetBuilder
from src.ml.dataset_inspect import inspect_manifest
from src.input.smart_capture import SmartCapturePolicy, SmartCaptureState, should_capture

def test_experience_id_and_summary():
    mem = ExperienceMemory(root=str(Path(tempfile.mkdtemp())))
    img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    s = mem.store(img, camera_source="test", detections=[{"class": "obstacle", "confidence": 0.5}],
                  capture_reason="MANUAL", free_space_ratio=0.5, risk_score=0.2, risk_level="LOW", decision="HOLD")
    assert s is not None and s.experience_id.startswith("EXP-MEM-")
    assert s.model_prediction == s.detections and s.human_annotation is None
    assert mem.summary()["total"] >= 1

def test_review_preserves_prediction():
    mem = ExperienceMemory(root=str(Path(tempfile.mkdtemp())))
    s = mem.store(np.zeros((32, 32, 3), dtype=np.uint8), detections=[{"class": "a", "confidence": 0.9}],
                  free_space_ratio=0, risk_score=0, risk_level="LOW", decision="GO", capture_reason="MANUAL")
    mem.set_review_status(s.experience_id, "corrected", human_annotation=[{"class": "b", "confidence": 1.0}])
    got = mem.get(s.experience_id)
    assert got["review_status"] == "corrected"
    assert got["model_prediction"][0]["class"] == "a"
    assert got["human_annotation"][0]["class"] == "b"

def test_dataset_from_accepted():
    root = Path(tempfile.mkdtemp())
    mem = ExperienceMemory(root=str(root / "exp"))
    for i in range(6):
        s = mem.store(np.full((32, 32, 3), i * 40, dtype=np.uint8),
                      detections=[{"class": "obstacle", "confidence": 0.8}],
                      free_space_ratio=0.5, risk_score=0.1, risk_level="LOW", decision="GO",
                      capture_reason="SMART_CAPTURE", skip_duplicate_hash=False)
        if s: mem.set_review_status(s.experience_id, "accepted")
    approved = [x for x in mem.list_samples(100) if x["review_status"] == "accepted"]
    man = DatasetBuilder(root=str(root / "ds")).build_from_experiences(approved, seed=7)
    assert man.sample_count >= 1 and isinstance(inspect_manifest(man.to_dict()), list)

def test_smart_capture_cooldown():
    pol = SmartCapturePolicy(uncertainty_threshold=0.2, cooldown_s=100.0)
    st = SmartCaptureState()
    assert should_capture(pol, st, [], uncertainty=0.9) is True
    assert should_capture(pol, st, [], uncertainty=0.9) is False
