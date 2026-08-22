from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..backbone.cnn import TinyConvBackbone
from ..heads.classification import ClassificationHead
from ..heads.detection import DetectionHead


class ARQTECHModel(nn.Module):
    """Experimental model architecture owned by this repository.

    The default forward path returns classification logits to preserve the
    existing synthetic bootstrap loop. Detection outputs are exposed through
    ``forward_all`` and must not be treated as validated detector predictions
    without reviewed detection annotations and evaluation.
    """

    model_name = "ARQTECH"
    model_version = "v0.2-modular"

    def __init__(self, num_classes: int = 4, in_channels: int = 3,
                 backbone_channels: tuple[int, ...] = (16, 32, 64),
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = TinyConvBackbone(in_channels=in_channels, channels=backbone_channels)
        self.classification_head = ClassificationHead(self.backbone.out_channels, self.num_classes, dropout=dropout)
        self.detection_head = DetectionHead(self.backbone.out_channels, self.num_classes)
        # Compatibility with the original ARQTechV01 implementation.
        self.head = self.classification_head

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.forward_features(x))

    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.forward_features(x)
        return {
            "features": features,
            "classification_logits": self.classification_head(features),
            **self.detection_head(features),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        outputs = self.forward_all(x)
        outputs["class_probabilities"] = outputs["classification_logits"].softmax(dim=-1)
        outputs["objectness_probability"] = outputs["objectness"].sigmoid()
        return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
