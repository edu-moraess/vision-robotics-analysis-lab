from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionDatasetConfig:
    dataset_root: str
    dataset_id: str
    split: str = "train"


@dataclass(frozen=True)
class DetectionDatasetValidation:
    valid: bool
    dataset_id: str
    sample_count: int
    class_names: tuple[str, ...]
    errors: tuple[str, ...]


class ReviewedDetectionDataset:
    """PyTorch-compatible dataset contract for real, human-reviewed detection data.

    The loader deliberately rejects incomplete or synthetic-only manifests. It does
    not generate labels from model predictions or external AI suggestions.
    """

    REQUIRED_SPLITS = ("train", "validation", "test")

    def __init__(self, config: DetectionDatasetConfig, transform: Optional[Any] = None):
        self.config = config
        self.transform = transform
        self.dataset_dir = Path(config.dataset_root) / config.dataset_id
        self.manifest: dict = {}
        self.sample_ids: list[str] = []
        self.class_names: tuple[str, ...] = ()
        self.validation = self.validate()
        if not self.validation.valid:
            raise ValueError("Invalid real detection dataset: " + "; ".join(self.validation.errors))
        self.manifest = json.loads((self.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        self.sample_ids = (self.dataset_dir / "splits" / f"{config.split}.txt").read_text(encoding="utf-8").splitlines()
        self.class_names = self.validation.class_names

    def validate(self) -> DetectionDatasetValidation:
        errors: list[str] = []
        class_names: set[str] = set()
        manifest_path = self.dataset_dir / "manifest.json"
        if not manifest_path.exists():
            return DetectionDatasetValidation(False, self.config.dataset_id, 0, (), ("manifest.json not found",))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return DetectionDatasetValidation(False, self.config.dataset_id, 0, (), (f"manifest unreadable: {type(exc).__name__}",))
        if manifest.get("dataset_id") != self.config.dataset_id:
            errors.append("manifest dataset_id mismatch")
        if manifest.get("annotation_status") != "HUMAN_VERIFIED":
            errors.append("annotation_status must be HUMAN_VERIFIED")
        if manifest.get("split_strategy") != "SOURCE_GROUPED":
            errors.append("split_strategy must be SOURCE_GROUPED")
        split_path = self.dataset_dir / "splits" / f"{self.config.split}.txt"
        if self.config.split not in self.REQUIRED_SPLITS:
            errors.append(f"unsupported split: {self.config.split}")
        if not split_path.exists():
            errors.append(f"split file not found: {split_path.name}")
            return DetectionDatasetValidation(False, self.config.dataset_id, 0, (), tuple(errors))
        sample_ids = split_path.read_text(encoding="utf-8").splitlines()
        if not sample_ids:
            errors.append(f"split {self.config.split} is empty")
        for sample_id in sample_ids:
            label_path = self.dataset_dir / "labels" / f"{sample_id}.json"
            if not label_path.exists():
                errors.append(f"missing label for {sample_id}")
                continue
            try:
                label = json.loads(label_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append(f"unreadable label for {sample_id}")
                continue
            if label.get("label_source") != "HUMAN_VERIFIED":
                errors.append(f"{sample_id}: label is not HUMAN_VERIFIED")
            if label.get("review_status") not in ("accepted", "corrected"):
                errors.append(f"{sample_id}: review_status is not accepted/corrected")
            image_path = self.dataset_dir / "images" / f"{sample_id}.jpg"
            if not image_path.exists():
                errors.append(f"missing image for {sample_id}")
            detections = label.get("detections")
            if not isinstance(detections, list):
                errors.append(f"{sample_id}: detections must be a list")
                continue
            for index, detection in enumerate(detections):
                if not isinstance(detection, dict):
                    errors.append(f"{sample_id}: detection {index} is not an object")
                    continue
                class_name = detection.get("class") or detection.get("class_name")
                if class_name is None:
                    errors.append(f"{sample_id}: detection {index} has no class")
                else:
                    class_names.add(str(class_name))
                bbox = detection.get("bbox") or detection.get("box")
                if not self._valid_bbox(bbox):
                    errors.append(f"{sample_id}: detection {index} requires numeric bbox [x, y, w, h]")
        if not class_names and not errors:
            errors.append("no detection classes found")
        return DetectionDatasetValidation(
            valid=not errors,
            dataset_id=self.config.dataset_id,
            sample_count=len(sample_ids),
            class_names=tuple(sorted(class_names)),
            errors=tuple(errors),
        )

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return False
        return all(math.isfinite(value) for value in values) and values[2] > 0 and values[3] > 0

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sample_id = self.sample_ids[index]
        image = Image.open(self.dataset_dir / "images" / f"{sample_id}.jpg").convert("RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        label = json.loads((self.dataset_dir / "labels" / f"{sample_id}.json").read_text(encoding="utf-8"))
        detections = label["detections"]
        target = {
            "boxes": np.asarray([d.get("bbox", d.get("box")) for d in detections], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray([
                self.class_names.index(str(d.get("class", d.get("class_name")))) for d in detections
            ], dtype=np.int64),
            "sample_id": sample_id,
        }
        item = {"image": image_array, "target": target}
        return self.transform(item) if self.transform else item
