from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, random_split

from .model_v01 import ARQTechV01, CLASS_NAMES, count_parameters
from .registry import register_train_result
from .status import ModelStatus
from .synthetic_data import SyntheticPatchDataset
from .training.engine import fit_classification


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
    lifecycle_status: str = ModelStatus.TRAINED.value
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def train_arqtech_v01(epochs=5, batch_size=32, lr=1e-3, n_samples=800, seed=42,
                      out_dir="data/models/arqtech_v01", device=None, registry_root=None) -> TrainResult:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_id = f"arqtech_v01_{uuid.uuid4().hex[:8]}"
    ds = SyntheticPatchDataset(n=n_samples, seed=seed)
    n_val = max(32, int(0.2 * len(ds)))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = ARQTechV01(num_classes=len(CLASS_NAMES)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fit = fit_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=optimizer,
        device=dev,
    )
    if fit.best_state is not None:
        model.load_state_dict(fit.best_state)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{exp_id}.pt"
    torch.save({
        "model": "ARQTECH",
        "version": "v0.2-modular",
        "experiment_id": exp_id,
        "class_names": CLASS_NAMES,
        "state_dict": fit.best_state,
        "input_size": 64,
        "task": "patch_classification",
        "dataset": "synthetic_patches",
        "lifecycle_status": ModelStatus.TRAINED.value,
        "notes": [
            "Bootstrap on synthetic patches only.",
            "NOT production detector.",
            "A checkpoint is not a validation claim.",
        ],
    }, ckpt)
    result = TrainResult(
        experiment_id=exp_id,
        model_name="ARQTECH",
        model_version="v0.2-modular",
        status="TRAINED_EXPERIMENTAL",
        lifecycle_status=ModelStatus.TRAINED.value,
        epochs_ran=fit.epochs_ran,
        train_loss=[round(x, 5) for x in fit.train_loss],
        val_loss=[round(x, 5) for x in fit.val_loss],
        train_acc=[round(x, 5) for x in fit.train_acc],
        val_acc=[round(x, 5) for x in fit.val_acc],
        best_val_acc=round(fit.best_val_acc, 5),
        best_val_loss=round(fit.best_val_loss, 5),
        parameters=count_parameters(model),
        device=str(dev),
        dataset="synthetic_patches",
        dataset_size=n_samples,
        duration_s=round(fit.duration_s, 3),
        checkpoint_path=str(ckpt),
        notes=[
            "Real PyTorch gradient descent executed.",
            "Validation accuracy is on the SYNTHETIC hold-out only.",
            "Do not report this as production object-detection mAP.",
            "Next: human-reviewed Experience Memory and reviewed detection data.",
        ],
    )
    (out / f"{exp_id}.json").write_text(json.dumps(result.to_dict(), indent=2))
    exp_dir = Path("data/experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    register_train_result(result.to_dict(), root=registry_root or str(out.parent))
    return result


# Forward-compatible generic name; the v0.1 alias remains for callers.
train_arqtech = train_arqtech_v01


if __name__ == "__main__":
    print(json.dumps(train_arqtech_v01(epochs=5).to_dict(), indent=2))
