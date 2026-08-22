from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..backbone.cnn import TinyConvBackbone
from ..heads.detection import DetectionHead


ARQTECH_V03_VERSION = "v0.3-detection-experimental"


@dataclass
class DetectionTaskConfig:
    version: str = ARQTECH_V03_VERSION
    task: str = "REAL OBJECT DETECTION"
    framework: str = "PyTorch"
    dataset_id: Optional[str] = None
    num_classes: int = 1
    input_resolution: tuple[int, int] = (640, 640)
    checkpoint_path: Optional[str] = None
    status: str = "EXPERIMENTAL"
    activation_status: str = "NOT AVAILABLE"
    notes: list[str] = field(default_factory=lambda: [
        "Requires a human-reviewed real detection dataset.",
        "Requires detection loss, box decode and post-processing validation.",
        "Never use the v0.2 synthetic classification checkpoint as a detector.",
    ])

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["input_resolution"] = list(self.input_resolution)
        return payload


class ARQTECHV03DetectionModel(nn.Module):
    """Research detection head for v0.3; raw outputs require a reviewed protocol."""

    model_name = "ARQTECH"
    model_version = ARQTECH_V03_VERSION
    task = "REAL OBJECT DETECTION"

    def __init__(self, num_classes: int = 1, in_channels: int = 3,
                 backbone_channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = TinyConvBackbone(in_channels=in_channels, channels=backbone_channels)
        self.detection_head = DetectionHead(self.backbone.out_channels, self.num_classes)

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        return self.detection_head(features)

    @torch.no_grad()
    def raw_predict(self, images: torch.Tensor) -> dict:
        self.eval()
        outputs = self.forward(images)
        return {
            **outputs,
            "objectness_probability": outputs["objectness"].sigmoid(),
            "class_probabilities": outputs["class_logits"].softmax(dim=-1),
            "output_status": "RAW EXPERIMENTAL HEAD OUTPUTS",
        }


class ARQTECHV03DetectorAdapter:
    """Activation gate for v0.3; it never silently promotes a checkpoint."""

    def __init__(self, config: Optional[DetectionTaskConfig] = None):
        self.config = config or DetectionTaskConfig()
        self.model = ARQTECHV03DetectionModel(num_classes=self.config.num_classes)
        self.load_error: Optional[str] = None
        self._loaded = False
        self._load_if_explicit_checkpoint()

    @property
    def identity(self) -> dict:
        return {
            "model": "ARQTECH",
            "version": self.config.version,
            "task": self.config.task,
            "framework": self.config.framework,
            "status": self.config.status,
            "activation_status": self.config.activation_status,
            "dataset": self.config.dataset_id or "NOT AVAILABLE",
            "checkpoint": self.config.checkpoint_path or "NOT AVAILABLE",
            "available": self.available,
            "role": "EXPERIMENTAL RESEARCH MODEL",
        }

    @property
    def available(self) -> bool:
        return bool(
            self._loaded
            and self.config.dataset_id
            and self.config.status == "VALIDATED"
            and self.config.activation_status == "ACTIVE"
        )

    def _load_if_explicit_checkpoint(self) -> None:
        path = self.config.checkpoint_path
        if not path:
            return
        checkpoint = Path(path)
        if not checkpoint.exists():
            self.load_error = "checkpoint not found"
            return
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            self.model.load_state_dict(state, strict=False)
            self._loaded = True
        except Exception as exc:
            self.load_error = type(exc).__name__

    def detect(self, frame):
        if not self.available:
            raise RuntimeError(
                "ARQTECH v0.3 detection is unavailable: reviewed dataset, validated status, "
                "active lifecycle and checkpoint are required."
            )
        raise NotImplementedError(
            "Validated box decoding/post-processing is not configured for this experimental adapter."
        )
