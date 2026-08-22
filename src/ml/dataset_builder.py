from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetManifest:
    dataset_id: str
    creation_time: float
    sample_count: int
    class_count: int
    class_distribution: Dict[str, int]
    train_count: int
    validation_count: int
    test_count: int
    source_distribution: Dict[str, int]
    annotation_status: str
    split_seed: int
    notes: List[str] = field(default_factory=list)
    split_strategy: str = "SOURCE_GROUPED"
    group_distribution: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    split_warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class DatasetBuilder:
    def __init__(self, root: str = "data/datasets"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _next_id(self):
        n = len(list(self.root.glob("dataset_v*"))) + 1
        return f"dataset_v{n:03d}"

    def build_from_experiences(self, experiences, train_ratio=0.7, val_ratio=0.15,
                               test_ratio=0.15, seed=42, dataset_id=None,
                               split_strategy="SOURCE_GROUPED"):
        approved = [e for e in experiences if e.get("review_status") in ("accepted", "corrected")]
        if not approved:
            raise ValueError("No accepted/corrected experiences to build a dataset.")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        ds_id = dataset_id or self._next_id()
        ds_dir = self.root / ds_id
        if ds_dir.exists():
            raise ValueError(f"Dataset {ds_id} already exists — versions are immutable.")
        for sub in ("images", "labels", "splits"):
            (ds_dir / sub).mkdir(parents=True)

        unique, duplicates_removed = self._deduplicate(approved)
        groups: Dict[str, List[dict]] = {}
        for item in unique:
            groups.setdefault(self._group_key(item), []).append(item)
        train_items, val_items, test_items = self._grouped_split(
            groups, train_ratio, val_ratio, test_ratio, seed,
        )
        class_dist, source_dist = {}, {}
        group_dist = {
            "train": len({self._group_key(e) for e in train_items}),
            "validation": len({self._group_key(e) for e in val_items}),
            "test": len({self._group_key(e) for e in test_items}),
        }
        split_warnings = []
        for split_name, ratio, subset in (("train", train_ratio, train_items), ("validation", val_ratio, val_items), ("test", test_ratio, test_items)):
            if ratio > 0 and not subset:
                split_warnings.append(
                    f"{split_name} split is empty because grouped splitting had too few independent source/session groups."
                )

        def _export(split_name, subset):
            ids = []
            for e in subset:
                sid = e.get("sample_id", f"unk_{len(ids)}")
                ids.append(sid)
                src_img = e.get("image_path")
                if src_img and Path(src_img).exists():
                    dest = ds_dir / "images" / f"{sid}.jpg"
                    if not dest.exists():
                        shutil.copy2(src_img, dest)
                review_status = e.get("review_status")
                human_annotation = e.get("human_annotation")
                annotations = human_annotation if review_status == "corrected" and human_annotation else e.get("detections", [])
                label = {
                    "sample_id": sid,
                    "review_status": review_status,
                    "detections": annotations,
                    "model_prediction": e.get("model_prediction", e.get("detections", [])),
                    "label_source": "HUMAN_VERIFIED",
                    "model_prediction_kept": review_status == "accepted",
                    "external_analysis": e.get("external_analysis"),
                    "source_identifier": e.get("source_identifier") or e.get("camera_source"),
                    "session_id": e.get("session_id") or e.get("source_identifier") or e.get("camera_source"),
                }
                (ds_dir / "labels" / f"{sid}.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
                for detection in annotations or []:
                    if isinstance(detection, dict):
                        c = detection.get("class") or detection.get("class_name") or "unknown"
                        class_dist[c] = class_dist.get(c, 0) + 1
                src = e.get("camera_source") or e.get("source_identifier") or "unknown"
                source_dist[src] = source_dist.get(src, 0) + 1
            (ds_dir / "splits" / f"{split_name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

        _export("train", train_items)
        _export("val", val_items)
        _export("test", test_items)
        (ds_dir / "splits" / "groups.json").write_text(json.dumps({
            self._group_key(e): split for split, subset in (
                ("train", train_items), ("validation", val_items), ("test", test_items)
            ) for e in subset
        }, indent=2), encoding="utf-8")
        manifest = DatasetManifest(
            dataset_id=ds_id, creation_time=time.time(), sample_count=len(unique),
            class_count=len(class_dist), class_distribution=class_dist,
            train_count=len(train_items), validation_count=len(val_items), test_count=len(test_items),
            source_distribution=source_dist, annotation_status="HUMAN_VERIFIED", split_seed=seed,
            notes=[
                "Immutable dataset version.",
                "Labels from human review only.",
                "Splits are grouped by source/session when metadata is available.",
                "Duplicate image hashes are removed before splitting when provided.",
                *split_warnings,
            ], split_strategy=split_strategy, group_distribution=group_dist,
            duplicates_removed=duplicates_removed, split_warnings=split_warnings,
        )
        (ds_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _group_key(experience: dict) -> str:
        return str(
            experience.get("session_id")
            or experience.get("source_identifier")
            or experience.get("camera_source")
            or experience.get("source_type")
            or "unknown-session"
        )

    @staticmethod
    def _deduplicate(experiences):
        seen = set()
        unique = []
        duplicates = 0
        for experience in experiences:
            image_hash = experience.get("image_hash")
            key = ("hash", image_hash) if image_hash else ("sample", experience.get("sample_id"))
            if key[1] and key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(experience)
        return unique, duplicates

    @staticmethod
    def _grouped_split(groups, train_ratio, val_ratio, test_ratio, seed):
        rng = random.Random(seed)
        group_items = list(groups.items())
        rng.shuffle(group_items)
        targets = {
            "train": sum(len(items) for _, items in group_items) * train_ratio,
            "validation": sum(len(items) for _, items in group_items) * val_ratio,
            "test": sum(len(items) for _, items in group_items) * test_ratio,
        }
        out = {"train": [], "validation": [], "test": []}
        counts = {key: 0 for key in out}
        for _, items in group_items:
            split = max(out, key=lambda key: (targets[key] - counts[key], targets[key]))
            out[split].extend(items)
            counts[split] += len(items)
        return out["train"], out["validation"], out["test"]

    def list_datasets(self):
        out = []
        for p in sorted(self.root.glob("dataset_v*/manifest.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        return out
