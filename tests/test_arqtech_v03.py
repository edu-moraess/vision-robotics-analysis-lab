import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arqtech.v03 import (
    ARQTECHV03DetectionModel,
    ARQTECHV03DetectorAdapter,
    DetectionDatasetConfig,
    DetectionTaskConfig,
    ReviewedDetectionDataset,
)
from src.ml.dataset_builder import DatasetBuilder


def test_v03_is_detection_head_and_not_v02_classifier():
    model = ARQTECHV03DetectionModel(num_classes=2)
    outputs = model(torch.zeros(1, 3, 64, 64))
    assert set(outputs) == {"objectness", "box", "class_logits"}
    assert model.model_version == "v0.3-detection-experimental"


def test_v03_adapter_refuses_unvalidated_activation():
    adapter = ARQTECHV03DetectorAdapter(DetectionTaskConfig())
    assert adapter.identity["available"] is False
    with pytest.raises(RuntimeError, match="reviewed dataset"):
        adapter.detect(None)


def test_reviewed_detection_dataset_requires_real_human_bboxes(tmp_path):
    source_image = tmp_path / "source.png"
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(source_image)
    manifest = DatasetBuilder(str(tmp_path / "datasets")).build_from_experiences([
        {
            "sample_id": "frame_001", "review_status": "corrected",
            "image_path": str(source_image), "image_hash": "hash-1",
            "source_identifier": "session-1",
            "human_annotation": [{"class": "person", "bbox": [1, 2, 4, 5]}],
            "detections": [{"class": "person", "bbox": [1, 2, 4, 5]}],
        },
    ], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    dataset = ReviewedDetectionDataset(DetectionDatasetConfig(
        dataset_root=str(tmp_path / "datasets"), dataset_id=manifest.dataset_id,
    ))
    assert len(dataset) == 1
    assert dataset[0]["target"]["boxes"].shape == (1, 4)


def test_reviewed_detection_dataset_rejects_prediction_only_label(tmp_path):
    root = tmp_path / "datasets" / "dataset_v001"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "splits").mkdir()
    (root / "splits" / "train.txt").write_text("frame_001\n")
    (root / "manifest.json").write_text(json.dumps({
        "dataset_id": "dataset_v001", "annotation_status": "HUMAN_VERIFIED",
        "split_strategy": "SOURCE_GROUPED",
    }))
    (root / "labels" / "frame_001.json").write_text(json.dumps({
        "review_status": "accepted", "label_source": "MODEL_PREDICTION",
        "detections": [{"class": "person", "bbox": [0, 0, 2, 2]}],
    }))
    with pytest.raises(ValueError, match="HUMAN_VERIFIED"):
        ReviewedDetectionDataset(DetectionDatasetConfig(
            dataset_root=str(tmp_path / "datasets"), dataset_id="dataset_v001",
        ))
