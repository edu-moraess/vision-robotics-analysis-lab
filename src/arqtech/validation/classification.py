from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import torch
from torch.utils.data import DataLoader


@dataclass
class ValidationResult:
    status: str
    dataset: str
    sample_count: int
    loss: float
    accuracy: float
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_classification(model, loader: DataLoader, device="cpu", dataset="unknown") -> ValidationResult:
    dev = torch.device(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(x.size(0))
    return ValidationResult(
        status="VALIDATED",
        dataset=dataset,
        sample_count=total,
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        notes=[
            "Classification validation only; lifecycle status is dataset-scoped.",
            "Results are valid for the supplied dataset and protocol only.",
            "This is not object-detection mAP, precision or recall.",
        ],
    )
