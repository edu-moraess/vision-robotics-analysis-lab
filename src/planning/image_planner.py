"""Image-space A* and Dijkstra. Paths are pixel trajectories — not metric."""
from __future__ import annotations
import heapq, math, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class ImagePlanResult:
    algorithm: str
    path_px: List[Tuple[int, int]]
    path_length_px: float
    nodes_explored: int
    execution_time_ms: float
    success: bool
    grid_shape: Tuple[int, int]
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"algorithm": self.algorithm, "path_length_px": round(self.path_length_px, 1),
                "nodes_explored": self.nodes_explored, "execution_time_ms": round(self.execution_time_ms, 2),
                "success": self.success, "grid_shape": self.grid_shape, "num_waypoints": len(self.path_px),
                "notes": self.notes}

def _heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _grid_search(occupancy, start, goal, use_heuristic=True):
    rows, cols = occupancy.shape
    sx, sy = start; gx, gy = goal
    if not (0 <= sx < cols and 0 <= sy < rows and 0 <= gx < cols and 0 <= gy < rows):
        return [], 0
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    costs = [math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)]
    open_set = []
    f0 = _heuristic((sx,sy),(gx,gy)) if use_heuristic else 0.0
    heapq.heappush(open_set, (f0, 0.0, sx, sy))
    came, gscore, closed, nodes = {}, {(sx,sy): 0.0}, set(), 0
    while open_set:
        f, g, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in closed:
            continue
        closed.add((cx, cy)); nodes += 1
        if cx == gx and cy == gy:
            path = [(cx, cy)]
            while (cx, cy) in came:
                cx, cy = came[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path, nodes
        for (dx, dy), cost in zip(neighbors, costs):
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < cols and 0 <= ny < rows) or occupancy[ny, nx] == 1:
                continue
            tg = g + cost
            if tg < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = tg
                came[(nx, ny)] = (cx, cy)
                nf = tg + (_heuristic((nx,ny),(gx,gy)) if use_heuristic else 0.0)
                heapq.heappush(open_set, (nf, tg, nx, ny))
    return [], nodes

class ImageSpacePlanner:
    def __init__(self, cell_size=16, inflation=1):
        self.cell_size = max(4, cell_size)
        self.inflation = inflation

    def plan(self, free_space_mask, algorithm="astar", start_px=None, goal_px=None):
        notes = ["Path computed in image-space (pixels).", "Not a metric navigation trajectory."]
        h, w = free_space_mask.shape[:2]
        grid_h, grid_w = max(1, h // self.cell_size), max(1, w // self.cell_size)
        occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for gy in range(grid_h):
            for gx in range(grid_w):
                y0, y1 = gy*self.cell_size, min(h, (gy+1)*self.cell_size)
                x0, x1 = gx*self.cell_size, min(w, (gx+1)*self.cell_size)
                cell = free_space_mask[y0:y1, x0:x1]
                if float(np.count_nonzero(cell)) / max(cell.size, 1) < 0.45:
                    occupancy[gy, gx] = 1
        if self.inflation > 0:
            try:
                from scipy import ndimage
                occupancy = ndimage.binary_dilation(occupancy, iterations=self.inflation).astype(np.uint8)
            except Exception:
                pass
        if start_px is None:
            start_px = (w//2, h - self.cell_size//2)
        if goal_px is None:
            goal_px = (w//2, self.cell_size)
        start_g = (max(0, min(grid_w-1, start_px[0]//self.cell_size)), max(0, min(grid_h-1, start_px[1]//self.cell_size)))
        goal_g = (max(0, min(grid_w-1, goal_px[0]//self.cell_size)), max(0, min(grid_h-1, goal_px[1]//self.cell_size)))
        occupancy[start_g[1], start_g[0]] = 0
        occupancy[goal_g[1], goal_g[0]] = 0
        t0 = time.perf_counter()
        path_g, nodes = _grid_search(occupancy, start_g, goal_g, algorithm.lower()=="astar")
        elapsed = (time.perf_counter()-t0)*1000.0
        path_px = []
        for gx, gy in path_g:
            path_px.append((min(w-1, int((gx+0.5)*self.cell_size)), min(h-1, int((gy+0.5)*self.cell_size))))
        length = sum(math.hypot(path_px[i][0]-path_px[i-1][0], path_px[i][1]-path_px[i-1][1]) for i in range(1, len(path_px)))
        return ImagePlanResult(algorithm.lower(), path_px, length, nodes, elapsed, len(path_px)>0, (grid_h, grid_w), notes)

    def compare(self, free_space_mask, start_px=None, goal_px=None):
        return [self.plan(free_space_mask, "astar", start_px, goal_px),
                self.plan(free_space_mask, "dijkstra", start_px, goal_px)]
