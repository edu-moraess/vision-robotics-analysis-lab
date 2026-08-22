from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..model.arqtech_model import ARQTECHModel


class ARQTECHInference:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None,
                 num_classes: int = 4):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path
        self.model = ARQTECHModel(num_classes=num_classes).to(self.device)
        self.loaded = False
        self.load_error = None
        if checkpoint_path:
            self.load(checkpoint_path)

    @property
    def identity(self) -> dict:
        return {
            "model": "ARQTECH",
            "model_type": "EXPERIMENTAL PYTORCH MODEL",
            "model_version": "v0.2-modular",
            "weights": self.checkpoint_path or "NONE",
            "status": "LOADED" if self.loaded else "NOT LOADED",
            "device": str(self.device),
            "error": self.load_error,
        }

    def load(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            self.load_error = f"checkpoint not found: {path}"
            return
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
            state = payload.get("state_dict") if isinstance(payload, dict) else payload
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.checkpoint_path = str(path)
            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> dict:
        if not self.loaded:
            raise RuntimeError("ARQTECH checkpoint is not loaded; model status is NOT LOADED")
        return self.model.predict(tensor.to(self.device))
