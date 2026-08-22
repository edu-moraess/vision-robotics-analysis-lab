from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np

from .occupancy import OccupancyGrid, build_cost_map


@dataclass
class NavigationCostMap:
    grid: np.ndarray
    cell_size: int
    bands: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "shape": tuple(int(v) for v in self.grid.shape),
            "cell_size": self.cell_size,
            "bands": dict(self.bands),
            "sources": list(self.sources),
            "notes": list(self.notes),
            "space": "IMAGE-SPACE / SIMULATION-SPACE",
            "metric_cost": False,
        }

    def to_array(self) -> np.ndarray:
        return self.grid


def build_navigation_cost_map(occupancy: OccupancyGrid, risk_assessment=None,
                              predicted_trajectories: Optional[Iterable[dict]] = None,
                              obstacle_cost: float = 1000.0) -> NavigationCostMap:
    cost = build_cost_map(occupancy, inflation=1, obstacle_cost=obstacle_cost)
    sources = ["OCCUPANCY"]
    bands = {"LOW": 1.0, "MEDIUM": 10.0, "HIGH": 100.0, "CRITICAL": obstacle_cost}
    h, w = cost.shape
    if risk_assessment is not None:
        sources.append("RISK")
        for zone in getattr(risk_assessment, "risk_zones", []) or []:
            bbox = zone.get("bbox")
            level = str(zone.get("level", "LOW"))
            if not bbox:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            gx1, gx2 = max(0, x1 // occupancy.cell_size), min(w - 1, max(0, (x2 - 1) // occupancy.cell_size))
            gy1, gy2 = max(0, y1 // occupancy.cell_size), min(h - 1, max(0, (y2 - 1) // occupancy.cell_size))
            for gy in range(gy1, gy2 + 1):
                for gx in range(gx1, gx2 + 1):
                    cost[gy, gx] = max(cost[gy, gx], bands.get(level, bands["MEDIUM"]))
    trajectory_count = 0
    for observation in predicted_trajectories or []:
        for point in observation.get("predicted_trajectory", []) or []:
            center = point.get("center") if isinstance(point, dict) else None
            if not center:
                continue
            gx = max(0, min(w - 1, int(float(center[0]) // occupancy.cell_size)))
            gy = max(0, min(h - 1, int(float(center[1]) // occupancy.cell_size)))
            cost[gy, gx] = max(cost[gy, gx], bands["HIGH"])
            trajectory_count += 1
    if trajectory_count:
        sources.append("PREDICTED TRAJECTORY")
    return NavigationCostMap(
        grid=cost, cell_size=occupancy.cell_size, bands=bands, sources=sources,
        notes=[
            "Cost map is an image-space planning aid.",
            "Risk and predicted trajectory overlays are deterministic evidence layers.",
            "No metric collision probability is implied.",
        ],
    )
