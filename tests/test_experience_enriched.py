import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learning.experience import ExperienceMemory


def test_experience_memory_persists_temporal_scene_fields(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "experience"))
    sample = memory.store(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        detections=[{"class_name": "person", "mask_available": True}],
        masks=[{"mask_available": True}],
        geometry=[{"geometry_source": "MASK_CONTOUR"}],
        motion=[{"track_id": 1, "motion_state": "MOVING"}],
        trajectories=[{"track_id": 1, "samples": []}],
        risk={"level": "LOW"}, occupancy={"space": "IMAGE-SPACE"},
        simulation={"status": "SIMULATION"}, skip_duplicate_hash=False,
    )
    assert sample is not None
    stored = memory.get(sample.experience_id)
    assert stored["masks"][0]["mask_available"] is True
    assert stored["motion"][0]["motion_state"] == "MOVING"
    assert stored["simulation"]["status"] == "SIMULATION"
