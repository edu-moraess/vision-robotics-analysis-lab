"""Image-space occupancy grid and cost map. NOT metric-world."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 1, 2

@dataclass
class OccupancyGrid:
    grid: np.ndarray
    cell_size: int
    image_shape: Tuple[int, int]
    label: str = "IMAGE-SPACE OCCUPANCY GRID"

    @property
    def shape(self):
        return self.grid.shape

    def free_ratio(self) -> float:
        return float(np.count_nonzero(self.grid == FREE)) / max(self.grid.size, 1)

def build_occupancy_from_mask(free_space_mask, cell_size=16, free_threshold=0.45) -> OccupancyGrid:
    h, w = free_space_mask.shape[:2]
    gh, gw = max(1, h // cell_size), max(1, w // cell_size)
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * cell_size, min(h, (gy + 1) * cell_size)
            x0, x1 = gx * cell_size, min(w, (gx + 1) * cell_size)
            cell = free_space_mask[y0:y1, x0:x1]
            ratio = float(np.count_nonzero(cell)) / max(cell.size, 1)
            grid[gy, gx] = FREE if ratio >= free_threshold else OCCUPIED
    return OccupancyGrid(grid=grid, cell_size=cell_size, image_shape=(h, w))

def build_cost_map(occupancy: OccupancyGrid, inflation=1, obstacle_cost=1000.0, proximity_weight=5.0):
    grid = occupancy.grid
    cost = np.ones(grid.shape, dtype=np.float32)
    obs = (grid == OCCUPIED).astype(np.uint8)
    if inflation > 0:
        try:
            from scipy import ndimage
            inflated = ndimage.binary_dilation(obs, iterations=inflation).astype(np.uint8)
            dist = ndimage.distance_transform_edt(1 - inflated)
            cost = 1.0 + proximity_weight / (dist + 1.0)
            cost[inflated == 1] = obstacle_cost
        except Exception:
            cost[obs == 1] = obstacle_cost
    else:
        cost[obs == 1] = obstacle_cost
    cost[grid == UNKNOWN] = obstacle_cost * 0.5
    return cost
