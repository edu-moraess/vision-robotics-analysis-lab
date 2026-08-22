from __future__ import annotations

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Grid-free bootstrap head; requires detection annotations before use as a detector."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.objectness = nn.Linear(in_features, 1)
        self.box = nn.Linear(in_features, 4)
        self.class_logits = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> dict:
        return {
            "objectness": self.objectness(features),
            "box": self.box(features),
            "class_logits": self.class_logits(features),
        }
