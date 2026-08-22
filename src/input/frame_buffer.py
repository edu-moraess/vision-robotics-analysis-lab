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
    discarded: int = 0

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "drops": self.drops,
            "pushes": self.pushes,
            "discarded": self.discarded,
        }


class FrameBuffer:
    def __init__(self, capacity: int = 3):
        self.capacity = max(1, int(capacity))
        self._q: Deque[FramePacket] = deque(maxlen=self.capacity)
        self.drops = 0
        self.pushes = 0
        self.discarded = 0

    def push(self, packet: Optional[FramePacket]):
        if packet is None:
            return
        if len(self._q) == self.capacity:
            self.drops += 1
        self._q.append(packet)
        self.pushes += 1

    def pop_latest(self) -> Optional[FramePacket]:
        if not self._q:
            return None
        pkt = self._q[-1]
        self.discarded += max(0, len(self._q) - 1)
        self._q.clear()
        return pkt

    def pop(self) -> Optional[FramePacket]:
        if not self._q:
            return None
        return self._q.popleft()

    def clear(self) -> None:
        self.discarded += len(self._q)
        self._q.clear()

    def stats(self) -> BufferStats:
        return BufferStats(len(self._q), self.capacity, self.drops, self.pushes, self.discarded)
