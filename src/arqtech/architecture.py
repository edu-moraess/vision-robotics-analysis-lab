"""ARQTECH modular architecture specification (research design, not trained weights)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class ArqtechConfig:
    version: str = "v0.0-scaffold"
    backbone: str = "unspecified"
    neck: str = "unspecified"
    detection_head: str = "unspecified"
    classification_head: str = "unspecified"
    segmentation_head: Optional[str] = None
    input_resolution: tuple = (640, 640)
    num_classes: int = 4
    parameter_count: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_resolution"] = list(self.input_resolution)
        return d

def describe_architecture() -> Dict[str, Any]:
    return {
        "name": "ARQTECH",
        "full_name": "Autonomous Robotics Perception Architecture",
        "status": "EXPERIMENTAL_SCAFFOLD",
        "trained": False,
        "modules_planned": [
            "Backbone", "Neck / Feature Aggregation", "Detection Head",
            "Classification Head", "Optional Segmentation Head",
            "Loss Functions", "Post Processing", "Inference Engine",
        ],
        "current_baseline_detector": "ClassicalDetector (OpenCV)",
        "yolo_baseline": "OPTIONAL — not bundled",
        "philosophy": [
            "Do not rename a third-party architecture as ARQTECH.",
            "Report metrics only from actual training/evaluation runs.",
            "If baseline outperforms ARQTECH, report that honestly.",
        ],
        "roadmap_phases": [
            "PHASE 1 — Classical / YOLO baseline",
            "PHASE 2 — Dataset infrastructure",
            "PHASE 3 — Experience Memory",
            "PHASE 4 — Active Learning + human review",
            "PHASE 5 — ARQTECH prototype architecture",
            "PHASE 6 — From-scratch training",
            "PHASE 7 — Benchmarking vs baseline",
            "PHASE 8 — Optimization",
            "PHASE 9 — Edge inference",
            "PHASE 10 — Robotics hardware integration",
        ],
    }
