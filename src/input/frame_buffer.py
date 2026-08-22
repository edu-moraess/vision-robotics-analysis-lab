from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
from ..camera.base import FramePacket

@dataclass
class BufferStats:
    size: int
    capacity: int
    drops: int
    pushes: int

class FrameBuffer:
    def __init__(self, capacity: int = 3):
        self.capacity = max(1, capacity)
        self._q: Deque[FramePacket] = deque(maxlen=self.capacity)
        self.drops = 0
        self.pushes = 0

    def push(self, packet):
        if packet is None: return
        if len(self._q) == self.capacity: self.drops += 1
        self._q.append(packet); self.pushes += 1

    def pop_latest(self):
        if not self._q: return None
        pkt = self._q[-1]; self._q.clear(); return pkt

    def stats(self):
        return BufferStats(len(self._q), self.capacity, self.drops, self.pushes)
