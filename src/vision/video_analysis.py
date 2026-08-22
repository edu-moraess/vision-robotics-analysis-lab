"""Recorded video analysis — same AnalysisPipeline, aggregate report/events."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
import numpy as np
from ..core.pipeline import AnalysisPipeline, AnalysisResult

@dataclass
class VideoEvent:
    timestamp_s: float
    frame_id: int
    event_type: str
    detail: str
    def to_dict(self):
        return asdict(self)

@dataclass
class VideoAnalysisReport:
    filename: str
    frames_analyzed: int
    frames_skipped: int
    duration_processed_s: float
    objects_total: int
    unique_classes: List[str]
    class_counts: Dict[str, int]
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    events: List[dict]
    track_spans: Dict[str, dict]
    processing_time_s: float
    notes: List[str] = field(default_factory=list)
    def to_dict(self):
        return asdict(self)

class VideoAnalyzer:
    def __init__(self, pipeline: Optional[AnalysisPipeline] = None):
        self.pipeline = pipeline or AnalysisPipeline(enable_tracking=True)

    def analyze_frame(self, image: np.ndarray, run_planner: bool = True) -> AnalysisResult:
        return self.pipeline.run(image, run_planner=run_planner)

    def build_report(self, filename, results, frame_ids, timestamps_s, frames_skipped, processing_time_s):
        confs, class_counts, events = [], Counter(), []
        seen_classes, track_spans, prev_classes = set(), {}, set()
        for res, fid, ts in zip(results, frame_ids, timestamps_s):
            cur = set()
            for d in res.detections:
                name = d.class_name
                cur.add(name)
                class_counts[name] += 1
                confs.append(float(d.confidence))
                if name not in seen_classes:
                    seen_classes.add(name)
                    events.append(VideoEvent(ts, fid, "NEW_CLASS", name))
                if name not in track_spans:
                    track_spans[name] = {"first_s": ts, "last_s": ts}
                else:
                    track_spans[name]["last_s"] = ts
            for c in cur - prev_classes:
                events.append(VideoEvent(ts, fid, "OBJECT_ENTERED", c))
            for c in prev_classes - cur:
                events.append(VideoEvent(ts, fid, "OBJECT_LEFT", c))
            if res.navigation_state:
                st = res.navigation_state.get("status")
                if st in ("REPLANNING", "OBSTACLE_DETECTED", "STOPPED"):
                    events.append(VideoEvent(ts, fid, st, res.navigation_state.get("message", "")))
            prev_classes = cur
        return VideoAnalysisReport(
            filename=filename, frames_analyzed=len(results), frames_skipped=frames_skipped,
            duration_processed_s=float(timestamps_s[-1] - timestamps_s[0]) if len(timestamps_s) > 1 else 0.0,
            objects_total=sum(class_counts.values()), unique_classes=sorted(class_counts.keys()),
            class_counts=dict(class_counts),
            avg_confidence=float(np.mean(confs)) if confs else 0.0,
            min_confidence=float(np.min(confs)) if confs else 0.0,
            max_confidence=float(np.max(confs)) if confs else 0.0,
            events=[e.to_dict() for e in events[:200]], track_spans=track_spans,
            processing_time_s=processing_time_s,
            notes=["Same AnalysisPipeline as live camera.", "Timestamps approx. from source FPS / frame index."],
        )
