"""Immutable dataset versions from human-reviewed experiences only."""
from __future__ import annotations
import json, random, shutil, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    def to_dict(self):
        return asdict(self)

class DatasetBuilder:
    def __init__(self, root: str = "data/datasets"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _next_id(self):
        n = len(list(self.root.glob("dataset_v*"))) + 1
        return f"dataset_v{n:03d}"

    def build_from_experiences(self, experiences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, dataset_id=None):
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
        rng = random.Random(seed)
        items = list(approved)
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio)) if n > 2 else n
        n_val = int(n * val_ratio) if n > 2 else 0
        train_items, val_items, test_items = items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]
        class_dist, source_dist = {}, {}

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
                    "model_prediction": e.get("detections", []),
                    "label_source": "HUMAN_VERIFIED",
                    "model_prediction_kept": review_status == "accepted",
                    "external_analysis": e.get("external_analysis"),
                }
                (ds_dir / "labels" / f"{sid}.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
                for d in e.get("detections") or []:
                    if isinstance(d, dict):
                        c = d.get("class") or d.get("class_name") or "unknown"
                        class_dist[c] = class_dist.get(c, 0) + 1
                src = e.get("camera_source") or "unknown"
                source_dist[src] = source_dist.get(src, 0) + 1
            (ds_dir / "splits" / f"{split_name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

        _export("train", train_items); _export("val", val_items); _export("test", test_items)
        manifest = DatasetManifest(
            dataset_id=ds_id, creation_time=time.time(), sample_count=len(approved),
            class_count=len(class_dist), class_distribution=class_dist,
            train_count=len(train_items), validation_count=len(val_items), test_count=len(test_items),
            source_distribution=source_dist, annotation_status="HUMAN_VERIFIED", split_seed=seed,
            notes=["Immutable dataset version.", "Labels from human review only."],
        )
        (ds_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return manifest

    def list_datasets(self):
        out = []
        for p in sorted(self.root.glob("dataset_v*/manifest.json")):
            try: out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError: pass
        return out
