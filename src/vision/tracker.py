"""Temporal IoU tracking with explicit lifecycle and image-space history."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    position_history: List[dict] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    raw_center: Optional[Tuple[float, float]] = None
    smoothed_center: Optional[Tuple[float, float]] = None
    measurement_available: bool = True
    source_model: str = "CURRENT DETECTOR"
    model_version: str = "baseline"
    class_id: Optional[int] = None
    motion_state: str = "UNKNOWN"
    motion: dict = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)
    predicted_trajectory: List[dict] = field(default_factory=list)

    def to_dict(self):
        raw = self.raw_center or self.center
        smooth = self.smoothed_center or self.center
        return {
            "id": self.track_id,
            "track_id": self.track_id,
            "class": self.class_name,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 3),
            "bbox": self.bbox,
            "center": (round(float(self.center[0]), 2), round(float(self.center[1]), 2)),
            "raw_center": (round(float(raw[0]), 2), round(float(raw[1]), 2)),
            "smoothed_center": (round(float(smooth[0]), 2), round(float(smooth[1]), 2)),
            "velocity": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
            "velocity_image_x": round(float(self.velocity[0]), 3),
            "velocity_image_y": round(float(self.velocity[1]), 3),
            "velocity_unit": "IMAGE-SPACE VELOCITY",
            "age": self.age,
            "hits": self.hits,
            "status": self.status,
            "time_since_update": self.time_since_update,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "source_model": self.source_model,
            "model_version": self.model_version,
            "position_history": list(self.position_history[-20:]),
            "motion_state": self.motion_state,
            "motion": dict(self.motion),
            "trajectory": list(self.trajectory[-20:]),
            "predicted_trajectory": list(self.predicted_trajectory),
        }


class IoUTracker:
    def __init__(self, max_age=12, min_hits=3, iou_threshold=0.3, temp_lost_age=3,
                 history_limit=120):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.temp_lost_age = int(temp_lost_age)
        self.history_limit = int(history_limit)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.active = False
        self._frame_index = 0
        self.events: List[dict] = []
        self.track_switches = 0

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.active = False
        self._frame_index = 0
        self.events = []
        self.track_switches = 0

    def update(self, detections: List[Detection], timestamp: Optional[float] = None,
               frame_id: Optional[int] = None) -> List[Track]:
        self._frame_index += 1
        current_frame = self._frame_index if frame_id is None else int(frame_id)
        self.events = []
        self.active = True

        # Motion prediction is intentionally image-space only.
        for tr in self.tracks.values():
            cx, cy = tr.center
            vx, vy = tr.velocity
            predicted = (cx + vx, cy + vy)
            tr.center = predicted
            tr.raw_center = predicted
            tr.smoothed_center = tr.smoothed_center or predicted
            tr.measurement_available = False
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
                    tr = self.tracks[tid]
                    iou = box_iou(det.bbox, tr.bbox)
                    if tr.class_name != det.class_name and iou < 0.55:
                        continue
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
                old_class = tr.class_name
                old = tr.center
                new = ((det.bbox[0] + det.bbox[2]) / 2.0,
                       (det.bbox[1] + det.bbox[3]) / 2.0)
                if old_class != det.class_name:
                    self.track_switches += 1
                tr.class_name = det.class_name
                tr.class_id = det.class_id
                tr.velocity = (new[0] - old[0], new[1] - old[1])
                tr.center = new
                tr.raw_center = new
                tr.smoothed_center = new
                tr.measurement_available = True
                tr.bbox = det.bbox
                tr.confidence = float(det.confidence)
                tr.source_model = det.source_model
                tr.model_version = det.model_version
                tr.confidence_history.append(tr.confidence)
                if len(tr.confidence_history) > 30:
                    tr.confidence_history = tr.confidence_history[-30:]
                tr.hits += 1
                tr.time_since_update = 0
                tr.last_seen_frame = current_frame
                tr.history.append(new)
                if len(tr.history) > self.history_limit:
                    tr.history = tr.history[-self.history_limit:]
                det.object_id = tid
                prev = tr.status
                tr.status = STATUS_CONFIRMED if tr.hits >= self.min_hits else STATUS_CANDIDATE
                if prev != tr.status and tr.status == STATUS_CONFIRMED:
                    self.events.append(self._event("OBJECT_CONFIRMED", tr, timestamp, current_frame))
                elif tr.status == STATUS_CONFIRMED:
                    self.events.append(self._event("OBJECT_UPDATED", tr, timestamp, current_frame))

        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            tid = self._next_id
            self._next_id += 1
            det.object_id = tid
            self.tracks[tid] = Track(
                track_id=tid,
                class_name=det.class_name,
                confidence=float(det.confidence),
                bbox=det.bbox,
                center=(cx, cy),
                raw_center=(cx, cy),
                smoothed_center=(cx, cy),
                confidence_history=[float(det.confidence)],
                history=[(cx, cy)],
                first_seen_frame=current_frame,
                last_seen_frame=current_frame,
                status=STATUS_CANDIDATE,
                source_model=det.source_model,
                model_version=det.model_version,
                class_id=det.class_id,
            )
            self.events.append(self._event("OBJECT_ENTERED", self.tracks[tid], timestamp, current_frame))

        to_delete = []
        for tid, tr in self.tracks.items():
            if tid in matched_trk:
                continue
            if tr.time_since_update >= self.max_age:
                tr.status = STATUS_LOST
                self.events.append(self._event("OBJECT_LEFT", tr, timestamp, current_frame))
                to_delete.append(tid)
            elif tr.time_since_update >= self.temp_lost_age:
                if tr.status != STATUS_TEMP_LOST:
                    tr.status = STATUS_TEMP_LOST
                    self.events.append(self._event("OBJECT_TEMPORARILY_LOST", tr, timestamp, current_frame))

        for tid in to_delete:
            del self.tracks[tid]

        self._record_positions(timestamp, current_frame)
        return [t for t in self.tracks.values() if t.status != STATUS_LOST]

    @staticmethod
    def _event(event_type, track, timestamp, frame_id):
        return {
            "event_type": event_type,
            "track_id": track.track_id,
            "class": track.class_name,
            "frame": frame_id,
            "timestamp": timestamp,
            "status": track.status,
        }

    def _record_positions(self, timestamp, frame_id):
        for tr in self.tracks.values():
            if tr.status not in (STATUS_CONFIRMED, STATUS_TEMP_LOST):
                continue
            raw = tr.raw_center or tr.center
            smooth = tr.smoothed_center or tr.center
            tr.position_history.append({
                "timestamp": timestamp,
                "frame_id": frame_id,
                "raw_x": round(float(raw[0]), 3),
                "raw_y": round(float(raw[1]), 3),
                "smooth_x": round(float(smooth[0]), 3),
                "smooth_y": round(float(smooth[1]), 3),
                "measurement_available": bool(tr.measurement_available),
                "velocity_image_x": round(float(tr.velocity[0]), 3),
                "velocity_image_y": round(float(tr.velocity[1]), 3),
                "velocity_unit": "IMAGE-SPACE VELOCITY",
            })
            if len(tr.position_history) > self.history_limit:
                tr.position_history = tr.position_history[-self.history_limit:]

    def confirmed_tracks(self):
        return [t for t in self.tracks.values() if t.status == STATUS_CONFIRMED]

    def stats(self):
        by = {}
        for t in self.tracks.values():
            by[t.status] = by.get(t.status, 0) + 1
        return {
            "active_tracks": len(self.tracks),
            "by_status": by,
            "frame_index": self._frame_index,
            "track_switches": self.track_switches,
        }
