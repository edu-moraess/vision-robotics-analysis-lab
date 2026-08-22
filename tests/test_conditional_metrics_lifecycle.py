import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arqtech.lifecycle import ARQTECH_LIFECYCLE, LifecycleRecord
from src.ml.metrics import ConditionalDetectionMetricEvaluator
from src.ml.training_config import TrainingConfig, save_training_config


def test_metrics_are_not_measured_without_real_inputs():
    result = ConditionalDetectionMetricEvaluator().evaluate()
    assert result.status == "NOT MEASURED"
    assert result.metrics["precision"] is None
    assert result.metrics["mAP@50"] is None


def test_metrics_are_measured_only_from_explicit_inputs():
    result = ConditionalDetectionMetricEvaluator().evaluate(
        predictions=[{"image_id": "a", "boxes": [[0, 0, 4, 4]], "labels": ["person"]}],
        ground_truth=[{"image_id": "a", "boxes": [[0, 0, 4, 4]], "labels": ["person"]}],
    )
    assert result.status == "MEASURED"
    assert result.metrics["precision"] == 1.0
    assert result.metrics["recall"] == 1.0
    assert result.metrics["mAP@50"] is None


def test_lifecycle_and_training_record_are_explicit(tmp_path):
    record = LifecycleRecord(status="REAL DATASET", dataset_id="dataset_v001")
    assert record.to_dict()["lifecycle"] == list(ARQTECH_LIFECYCLE)
    path = save_training_config(TrainingConfig(
        experiment_id="exp_v03", model_name="ARQTECH", training_mode="detection",
        dataset_id="dataset_v001", model_version="v0.3-detection-experimental",
    ), str(tmp_path))
    payload = json.loads(path.read_text())
    assert payload["lifecycle_status"] == "ARCHITECTURE"
    assert payload["metrics"]["train_loss"] is None
    assert payload["metrics"]["validation_loss"] is None
    assert payload["metrics"]["device"] is None
