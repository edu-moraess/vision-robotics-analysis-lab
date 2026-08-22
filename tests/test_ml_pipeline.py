from __future__ import annotations
import sys, tempfile, json
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ml.dataset_builder import DatasetBuilder
from src.ml.active_learning import rank_for_review
from src.ml.report import LearningReportGenerator
from src.ml.training_config import TrainingConfig, save_training_config
from src.arqtech.registry import register_train_result, ModelRegistry

def test_rank_for_review():
    samples = [
        {"sample_id": "a", "review_status": "pending", "uncertainty_overall": 0.9, "detections": []},
        {"sample_id": "b", "review_status": "accepted", "uncertainty_overall": 0.9, "detections": []},
        {"sample_id": "c", "review_status": "pending", "uncertainty_overall": 0.1, "detections": [{"confidence": 0.95}]},
    ]
    ranked = rank_for_review(samples, limit=5)
    assert all(r["review_status"] == "pending" for r in ranked)
    assert ranked[0]["sample_id"] == "a"

def test_dataset_builder_requires_approved():
    b = DatasetBuilder(root=str(Path(tempfile.mkdtemp())))
    try:
        b.build_from_experiences([{"review_status": "pending"}])
        assert False
    except ValueError:
        pass

def test_dataset_builder_and_report():
    root = Path(tempfile.mkdtemp())
    img_dir = root / "exp_images"; img_dir.mkdir()
    path = img_dir / "s1.jpg"
    cv2.imwrite(str(path), np.zeros((40, 40, 3), dtype=np.uint8))
    experiences = [{"sample_id": f"s{i}", "review_status": "accepted", "image_path": str(path),
                    "detections": [{"class": "obstacle", "confidence": 0.8}], "camera_source": "test"} for i in range(5)]
    b = DatasetBuilder(root=str(root / "datasets"))
    m = b.build_from_experiences(experiences, seed=1)
    assert m.sample_count == 5
    gen = LearningReportGenerator(reports_root=str(root / "reports"), datasets_root=str(root / "datasets"), registry_root=str(root / "empty_reg"))
    report = gen.generate(dataset_id=m.dataset_id, experience_samples=experiences)
    assert report["model_summary"]["metrics_not_measured"] or report["conclusion"]
    assert Path(report["export_json"]).exists()

def test_registry_keeps_execution_fields_empty_until_supplied(tmp_path):
    register_train_result({
        "model_name": "ARQTECH", "model_version": "v0.3-detection-experimental",
        "dataset_id": None, "status": "NOT TRAINED",
    }, root=str(tmp_path))
    row = ModelRegistry(root=str(tmp_path)).get("v0.3-detection-experimental")
    assert row["train_loss"] is None
    assert row["validation_loss"] is None
    assert row["learning_rate"] is None
    assert row["device"] is None
    assert row["metrics"]["mAP@50"] is None


def test_training_config_not_started():
    root = Path(tempfile.mkdtemp())
    cfg = TrainingConfig(experiment_id="exp_test", model_name="ARQTECH", training_mode="FROM_SCRATCH", dataset_id="dataset_v001")
    data = json.loads(save_training_config(cfg, root=str(root)).read_text())
    assert data["status"] == "CONFIGURED_NOT_STARTED"
    assert data["metrics"]["train_loss"] is None
    assert data["metrics"]["validation_loss"] is None
    assert data["metrics"]["mAP@50"] is None
