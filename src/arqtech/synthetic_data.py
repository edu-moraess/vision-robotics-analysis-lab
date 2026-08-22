"""Synthetic patch dataset for ARQTECH v0.1 bootstrap training."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

def _make_patch(label: int, size: int = 64, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if label == 0:
        img[:] = rng.integers(20, 60, size=(size, size, 3), dtype=np.uint8)
    elif label == 1:
        img[:] = rng.integers(40, 90, size=(size, size, 3), dtype=np.uint8)
        x0, y0 = rng.integers(5, 25), rng.integers(5, 25)
        x1, y1 = rng.integers(35, 60), rng.integers(35, 60)
        img[y0:y1, x0:x1] = rng.integers(10, 40, size=3)
    elif label == 2:
        img[:] = (120, 110, 100)
        for y in range(0, size, 8):
            img[y:y+3, :] = (80, 75, 70)
    else:
        img[:] = rng.integers(30, 70, size=(size, size, 3), dtype=np.uint8)
        cx = size // 2
        img[20:55, cx-8:cx+8] = (40, 60, 160)
        img[8:20, cx-6:cx+6] = (200, 170, 150)
    noise = rng.integers(-15, 15, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

class SyntheticPatchDataset(Dataset):
    def __init__(self, n: int = 400, size: int = 64, seed: int = 42):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.labels = self.rng.integers(0, 4, size=n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        y = int(self.labels[idx])
        img = _make_patch(y, self.size, self.rng)
        x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return x, y
