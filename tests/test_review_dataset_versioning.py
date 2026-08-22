import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learning.experience import ExperienceMemory
from src.ml.dataset_builder import DatasetBuilder


def test_human_review_actions_are_audited(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "experience"))
    sample = memory.store(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        detections=[{"class": "person", "bbox": [0, 0, 4, 4]}],
        skip_duplicate_hash=False,
    )
    assert memory.apply_review(sample.experience_id, "CHANGE CLASS", [{"class": "vehicle"}], reviewer="tester")
    row = memory.get(sample.experience_id)
    assert row["review_status"] == "corrected"
    assert row["human_annotation"][0]["class"] == "vehicle"
    assert row["review_history"][-1]["action"] == "CHANGE_CLASS"
    assert memory.apply_review(sample.experience_id, "DELETE", reviewer="tester")
    assert memory.get(sample.experience_id)["human_annotation"] == []


def test_dataset_split_is_grouped_by_source(tmp_path):
    experiences = []
    for source in ("video_a", "video_b", "video_c"):
        for index in range(2):
            experiences.append({
                "sample_id": f"{source}_{index}", "review_status": "accepted",
                "camera_source": source, "source_identifier": source,
                "detections": [{"class": "person", "confidence": 0.9}],
            })
    manifest = DatasetBuilder(str(tmp_path / "datasets")).build_from_experiences(
        experiences, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=4,
    )
    assert manifest.split_strategy == "SOURCE_GROUPED"
    dataset_dir = tmp_path / "datasets" / manifest.dataset_id
    split_sources = {}
    for split, filename in (("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")):
        for sample_id in (dataset_dir / "splits" / filename).read_text().splitlines():
            split_sources.setdefault(sample_id.rsplit("_", 1)[0], set()).add(split)
    assert all(len(splits) == 1 for splits in split_sources.values())


def test_dataset_manifest_warns_when_grouped_split_is_empty(tmp_path):
    experiences = [{
        "sample_id": "single", "review_status": "accepted",
        "source_identifier": "only-session",
        "detections": [{"class": "person", "bbox": [0, 0, 2, 2]}],
    }]
    manifest = DatasetBuilder(str(tmp_path / "datasets")).build_from_experiences(
        experiences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    )
    assert manifest.split_warnings
    assert any("empty" in warning for warning in manifest.split_warnings)
