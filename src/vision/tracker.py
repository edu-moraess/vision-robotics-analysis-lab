"""Baseline IoU multi-object tracker. Temporal data only — never invent trajectories from a single image."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .detector import Detection
from .geometry import box_iou

@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self):
        return {"id": self.track_id, "class": self.class_name, "confidence": round(self.confidence, 3),
                "bbox": self.bbox, "center": self.center,
                "velocity": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
                "age": self.age, "hits": self.hits}

class IoUTracker:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.active = False

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.active = False

    def update(self, detections: List[Detection]) -> List[Track]:
        for tr in self.tracks.values():
            cx, cy = tr.center
            vx, vy = tr.velocity
            tr.center = (cx + vx, cy + vy)
            x1, y1, x2, y2 = tr.bbox
            tr.bbox = (int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy))
            tr.time_since_update += 1
            tr.age += 1
        matched_det, matched_trk = set(), set()
        if detections and self.tracks:
            track_ids = list(self.tracks.keys())
            pairs = []
            for di, det in enumerate(detections):
                for tid in track_ids:
                    iou = box_iou(det.bbox, self.tracks[tid].bbox)
                    if iou >= self.iou_threshold:
                        pairs.append((iou, di, tid))
            pairs.sort(reverse=True)
            for iou, di, tid in pairs:
                if di in matched_det or tid in matched_trk:
                    continue
                matched_det.add(di)
                matched_trk.add(tid)
                det = detections[di]
                tr = self.tracks[tid]
                old, new = tr.center, det.center
                alpha = 0.6
                vx = alpha * (new[0] - old[0]) + (1 - alpha) * tr.velocity[0]
                vy = alpha * (new[1] - old[1]) + (1 - alpha) * tr.velocity[1]
                tr.velocity = (vx, vy)
                tr.bbox, tr.center = det.bbox, det.center
                tr.confidence, tr.class_name = det.confidence, det.class_name
                tr.hits += 1
                tr.time_since_update = 0
                tr.history.append(det.center)
                if len(tr.history) > 40:
                    tr.history = tr.history[-40:]
                det.object_id = tid
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, det.class_name, det.confidence, det.bbox, det.center, history=[det.center])
            det.object_id = tid
        for tid in [t for t, tr in self.tracks.items() if tr.time_since_update > self.max_age]:
            del self.tracks[tid]
        if any(tr.age > 1 for tr in self.tracks.values()):
            self.active = True
        return [tr for tr in self.tracks.values() if tr.hits >= self.min_hits or tr.age < 3]
