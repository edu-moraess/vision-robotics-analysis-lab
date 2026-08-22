from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ArqtechConfig:
    version: str = "v0.2-modular"
    framework: str = "PyTorch"
    backbone: str = "TinyConvBackbone"
    neck: str = "Feature vector aggregation"
    detection_head: str = "DetectionHead (experimental, annotation-dependent)"
    classification_head: str = "ClassificationHead"
    segmentation_head: Optional[str] = None
    input_resolution: tuple = (64, 64)
    num_classes: int = 4
    parameter_count: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_resolution"] = list(self.input_resolution)
        return d


def describe_architecture(model_status: str = "NOT TRAINED", checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    status = str(model_status or "NOT TRAINED").upper()
    trained = status in ("TRAINED", "TRAINED_EXPERIMENTAL", "VALIDATING", "VALIDATED", "ACTIVE")
    return {
        "name": "ARQTECH",
        "full_name": "Autonomous Robotics Perception Architecture",
        "framework": "PyTorch",
        "version": "v0.2-modular",
        "status": status,
        "trained": trained,
        "checkpoint": checkpoint_path or "NOT AVAILABLE",
        "modules": {
            "model": "src/arqtech/model/arqtech_model.py",
            "backbone": "src/arqtech/backbone/cnn.py",
            "heads": ["src/arqtech/heads/classification.py", "src/arqtech/heads/detection.py"],
            "loss": "src/arqtech/loss/classification.py",
            "training": "src/arqtech/training/engine.py",
            "validation": "src/arqtech/validation/classification.py",
            "inference": "src/arqtech/inference/engine.py",
        },
        "current_training_scope": "Synthetic patch classification bootstrap only",
        "current_baseline_detector": "CURRENT DETECTOR / OpenCV",
        "yolo_baseline": "OPTIONAL EXTERNAL NEURAL BASELINE — not ARQTECH",
        "groq_layer": "OPTIONAL EXTERNAL MULTIMODAL ANALYSIS — not ARQTECH",
        "limitations": [
            "No claim of production object detection.",
            "No claim of mAP, precision or recall without reviewed detection data.",
            "No automatic conversion of predictions into ground truth.",
        ],
        "roadmap_phases": [
            "PHASE 1 — Classical / YOLO baseline",
            "PHASE 2 — Dataset infrastructure",
            "PHASE 3 — Experience Memory",
            "PHASE 4 — Active Learning + human review",
            "PHASE 5 — Modular ARQTECH architecture",
            "PHASE 6 — From-scratch training",
            "PHASE 7 — Validation and registry",
            "PHASE 8 — Benchmarking vs external baseline",
            "PHASE 9 — Edge inference",
            "PHASE 10 — Robotics hardware integration",
        ],
    }
