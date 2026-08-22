"""IoU multi-object tracker with explicit track lifecycle.
Statuses: CANDIDATE → CONFIRMED → TEMPORARILY_LOST → LOST
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .detector import Detection
from .geometry import box_iou

STATUS_CANDIDATE = "CANDIDATE"
STATUS_CONFIRMED = "CONFIRMED"
STATUS_TEMP_LOST = "TEMPORARILY_LOST"
STATUS_LOST = "LOST"

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
    status: str = STATUS_CANDIDATE
    confidence_history: List[float] = field(default_factory=list)
    history: List[Tuple[float, float]] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0

    def to_dict(self):
        return {
            "id": self.track_id, "class": self.class_name,
            "confidence": round(self.confidence, 3), "bbox": self.bbox,
            "center": self.center,
            "velocity": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
            "age": self.age, "hits": self.hits, "status": self.status,
            "time_since_update": self.time_since_update,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
        }

class IoUTracker:
    def __init__(self, max_age=12, min_hits=3, iou_threshold=0.3, temp_lost_age=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.temp_lost_age = temp_lost_age
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.active = False
        self._frame_index = 0
        self.events: List[dict] = []

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.active = False
        self._frame_index = 0
        self.events = []

    def update(self, detections: List[Detection]) -> List[Track]:
        self._frame_index += 1
        self.events = []
        self.active = True
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
                    if self.tracks[tid].class_name != det.class_name and iou < 0.55:
                        continue
                    if iou >= self.iou_threshold:
                        pairs.append((iou, di, tid))
            pairs.sort(reverse=True)
            for iou, di, tid in pairs:
                if di in matched_det or tid in matched_trk:
                    continue
                matched_det.add(di); matched_trk.add(tid)
                det = detections[di]; tr = self.tracks[tid]
                old = tr.center
                new = ((det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0)
                tr.velocity = (new[0] - old[0], new[1] - old[1])
                tr.center = new; tr.bbox = det.bbox
                tr.confidence = float(det.confidence)
                tr.confidence_history.append(tr.confidence)
                if len(tr.confidence_history) > 30:
                    tr.confidence_history = tr.confidence_history[-30:]
                tr.hits += 1; tr.time_since_update = 0
                tr.last_seen_frame = self._frame_index
                tr.history.append(new)
                if len(tr.history) > 40:
                    tr.history = tr.history[-40:]
                prev = tr.status
                tr.status = STATUS_CONFIRMED if tr.hits >= self.min_hits else STATUS_CANDIDATE
                if prev != tr.status and tr.status == STATUS_CONFIRMED:
                    self.events.append({"event_type": "TRACK_CONFIRMED", "track_id": tr.track_id,
                                        "class": tr.class_name, "frame": self._frame_index})
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            tid = self._next_id; self._next_id += 1
            self.tracks[tid] = Track(
                track_id=tid, class_name=det.class_name, confidence=float(det.confidence),
                bbox=det.bbox, center=(cx, cy), confidence_history=[float(det.confidence)],
                history=[(cx, cy)], first_seen_frame=self._frame_index,
                last_seen_frame=self._frame_index, status=STATUS_CANDIDATE,
            )
            self.events.append({"event_type": "OBJECT_ENTERED", "track_id": tid,
                                "class": det.class_name, "frame": self._frame_index})
        to_delete = []
        for tid, tr in self.tracks.items():
            if tid in matched_trk:
                continue
            if tr.time_since_update >= self.max_age:
                tr.status = STATUS_LOST
                self.events.append({"event_type": "OBJECT_LEFT", "track_id": tid,
                                    "class": tr.class_name, "frame": self._frame_index})
                to_delete.append(tid)
            elif tr.time_since_update >= self.temp_lost_age:
                if tr.status != STATUS_TEMP_LOST:
                    tr.status = STATUS_TEMP_LOST
                    self.events.append({"event_type": "TRACK_TEMPORARILY_LOST", "track_id": tid,
                                        "class": tr.class_name, "frame": self._frame_index})
        for tid in to_delete:
            del self.tracks[tid]
        return [t for t in self.tracks.values() if t.status != STATUS_LOST]

    def confirmed_tracks(self):
        return [t for t in self.tracks.values() if t.status == STATUS_CONFIRMED]

    def stats(self):
        by = {}
        for t in self.tracks.values():
            by[t.status] = by.get(t.status, 0) + 1
        return {"active_tracks": len(self.tracks), "by_status": by, "frame_index": self._frame_index}
