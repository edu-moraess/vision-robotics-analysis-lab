"""ARQTECH v0.1 training loop — real optimization, measured losses."""
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .model_v01 import ARQTechV01, CLASS_NAMES, count_parameters
from .synthetic_data import SyntheticPatchDataset

@dataclass
class TrainResult:
    experiment_id: str
    model_name: str
    model_version: str
    status: str
    epochs_ran: int
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    best_val_acc: float
    best_val_loss: float
    parameters: int
    device: str
    dataset: str
    dataset_size: int
    duration_s: float
    checkpoint_path: Optional[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

def train_arqtech_v01(epochs=5, batch_size=32, lr=1e-3, n_samples=800, seed=42,
                      out_dir="data/models/arqtech_v01", device=None) -> TrainResult:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_id = f"arqtech_v01_{uuid.uuid4().hex[:8]}"
    ds = SyntheticPatchDataset(n=n_samples, seed=seed)
    n_val = max(32, int(0.2 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = ARQTechV01(num_classes=len(CLASS_NAMES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_val_loss, best_state = 0.0, float("inf"), None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)
        train_losses.append(total_loss / max(total, 1))
        train_accs.append(correct / max(total, 1))
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = crit(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                correct += int((logits.argmax(1) == y).sum().item())
                total += x.size(0)
        va_loss = total_loss / max(total, 1)
        va_acc = correct / max(total, 1)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        if va_acc >= best_val_acc:
            best_val_acc, best_val_loss = va_acc, va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{exp_id}.pt"
    torch.save({
        "model": "ARQTECH", "version": "v0.1", "experiment_id": exp_id,
        "class_names": CLASS_NAMES, "state_dict": best_state,
        "input_size": 64, "task": "patch_classification", "dataset": "synthetic_patches",
        "notes": ["Bootstrap on synthetic patches only.", "NOT production detector."],
    }, ckpt)
    result = TrainResult(
        experiment_id=exp_id, model_name="ARQTECH", model_version="v0.1",
        status="TRAINED_EXPERIMENTAL", epochs_ran=epochs,
        train_loss=[round(x, 5) for x in train_losses],
        val_loss=[round(x, 5) for x in val_losses],
        train_acc=[round(x, 5) for x in train_accs],
        val_acc=[round(x, 5) for x in val_accs],
        best_val_acc=round(best_val_acc, 5), best_val_loss=round(best_val_loss, 5),
        parameters=count_parameters(model), device=str(dev),
        dataset="synthetic_patches", dataset_size=n_samples,
        duration_s=round(time.perf_counter() - t0, 3), checkpoint_path=str(ckpt),
        notes=["Real gradient descent executed.", "Val accuracy on SYNTHETIC hold-out only.",
               "Do not report as production mAP.", "Next: human-reviewed Experience Memory."],
    )
    (out / f"{exp_id}.json").write_text(json.dumps(result.to_dict(), indent=2))
    exp_dir = Path("data/experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result

if __name__ == "__main__":
    print(json.dumps(train_arqtech_v01(epochs=5).to_dict(), indent=2))
