from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvBackbone(nn.Module):
    """Small measurable backbone for bootstrap experiments, not a production claim."""

    def __init__(self, in_channels: int = 3, channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        blocks = []
        previous = in_channels
        for current in channels:
            blocks.extend([
                nn.Conv2d(previous, current, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(current),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            previous = current
        blocks.append(nn.AdaptiveAvgPool2d(1))
        self.network = nn.Sequential(*blocks)
        self.out_channels = previous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).flatten(1)
