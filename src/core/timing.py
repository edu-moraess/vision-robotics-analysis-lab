"""Performance measurement helpers. Uses time.perf_counter only."""
from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator

@dataclass
class LatencyBreakdown:
    stages: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, ms: float) -> None:
        self.stages[name] = float(ms)

    @property
    def total_ms(self) -> float:
        return float(sum(self.stages.values()))

    def to_dict(self) -> Dict[str, float]:
        out = {k: round(v, 3) for k, v in self.stages.items()}
        out["total_ms"] = round(self.total_ms, 3)
        return out

@contextmanager
def measure(name: str, breakdown: LatencyBreakdown) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        breakdown.record(name, (time.perf_counter() - t0) * 1000.0)
