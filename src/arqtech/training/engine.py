from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class FitResult:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_val_loss: float = float("inf")
    epochs_ran: int = 0
    duration_s: float = 0.0
    best_state: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "epochs_ran": self.epochs_ran,
            "duration_s": self.duration_s,
        }


def fit_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: Callable = nn.CrossEntropyLoss(),
    device: str | torch.device = "cpu",
) -> FitResult:
    dev = torch.device(device)
    result = FitResult()
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * x.size(0)
            train_correct += int((logits.argmax(1) == y).sum().item())
            train_total += int(x.size(0))
        result.train_loss.append(train_loss / max(train_total, 1))
        result.train_acc.append(train_correct / max(train_total, 1))

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += float(loss.item()) * x.size(0)
                val_correct += int((logits.argmax(1) == y).sum().item())
                val_total += int(x.size(0))
        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        result.val_loss.append(epoch_val_loss)
        result.val_acc.append(epoch_val_acc)
        result.epochs_ran = epoch + 1
        if epoch_val_acc >= result.best_val_acc:
            result.best_val_acc = epoch_val_acc
            result.best_val_loss = epoch_val_loss
            result.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result.duration_s = time.perf_counter() - started
    return result
