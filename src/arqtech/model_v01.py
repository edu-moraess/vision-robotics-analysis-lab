from __future__ import annotations

from typing import List

from .model.arqtech_model import ARQTECHModel, count_parameters


CLASS_NAMES: List[str] = ["background", "obstacle", "wall", "person"]

# Backward-compatible name used by the original v0.1 training loop.
ARQTechV01 = ARQTECHModel

__all__ = ["ARQTechV01", "ARQTECHModel", "CLASS_NAMES", "count_parameters"]
