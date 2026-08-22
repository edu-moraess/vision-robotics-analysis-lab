"""ARQTECH v0.1 — tiny experimental CNN (from-scratch prototype)."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

CLASS_NAMES: List[str] = ["background", "obstacle", "wall", "person"]

class ARQTechV01(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
