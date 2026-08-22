from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

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
    model_identities: List[dict] = field(default_factory=list)
    unique_tracks: int = 0
    track_persistence: Dict[str, int] = field(default_factory=dict)
    person_track_stability: object = "N/A"
    duplicate_detections: int = 0
    stopped_transitions: int = 0
    path_blocked_transitions: int = 0
    replanning_events: int = 0
    track_switches: int = 0

    def to_dict(self):
        return asdict(self)


class VideoAnalyzer:
    def __init__(self, pipeline: Optional[AnalysisPipeline] = None):
        self.pipeline = pipeline or AnalysisPipeline(enable_tracking=True)

    def analyze_frame(self, image: np.ndarray, run_planner: bool = True,
                      timestamp: Optional[float] = None, frame_id: Optional[int] = None,
                      source: str = "recorded_video") -> AnalysisResult:
        return self.pipeline.run(
            image, run_planner=run_planner, timestamp=timestamp,
            frame_id=frame_id, source=source,
        )

    def build_report(self, filename, results, frame_ids, timestamps_s, frames_skipped, processing_time_s):
        confs, class_counts, events = [], Counter(), []
        seen_classes, track_spans, prev_classes = set(), {}, set()
        track_persistence = Counter()
        person_track_frames = Counter()
        person_frames = 0
        model_identities = []
        planner_events = Counter()
        duplicate_detections = 0
        track_switches = 0

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
            for track in (res.tracks or []):
                tid = str(track.track_id)
                track_persistence[tid] += 1
                if track.class_name == "person":
                    person_frames += 1
                    person_track_frames[tid] += 1
                if tid not in track_spans:
                    track_spans[tid] = {"first_s": ts, "last_s": ts, "track_id": track.track_id}
                else:
                    track_spans[tid]["last_s"] = ts
            for c in cur - prev_classes:
                events.append(VideoEvent(ts, fid, "OBJECT_ENTERED", c))
            for c in prev_classes - cur:
                events.append(VideoEvent(ts, fid, "OBJECT_LEFT", c))
            for event in (res.track_events or []):
                events.append(VideoEvent(ts, fid, event.get("event_type", "TRACK_EVENT"), str(event)))
            if res.navigation_state:
                state = res.navigation_state.get("status")
                if state in ("REPLANNING", "OBSTACLE_DETECTED", "STOPPED", "PATH_BLOCKED"):
                    planner_events[state] += 1
                    events.append(VideoEvent(ts, fid, state, res.navigation_state.get("message", "")))
            if res.fusion_stats:
                duplicate_detections += int(res.fusion_stats.get("duplicate_reduction", 0) or 0)
            track_switches += int((res.telemetry or {}).get("track_switches", 0) or 0)
            identity = res.model_identity or {}
            if identity and identity not in model_identities:
                model_identities.append(identity)
            prev_classes = cur

        person_stability = "N/A"
        if person_frames:
            person_stability = round(max(person_track_frames.values()) / float(person_frames), 3)
        return VideoAnalysisReport(
            filename=filename, frames_analyzed=len(results), frames_skipped=frames_skipped,
            duration_processed_s=float(timestamps_s[-1] - timestamps_s[0]) if len(timestamps_s) > 1 else 0.0,
            objects_total=sum(class_counts.values()), unique_classes=sorted(class_counts.keys()),
            class_counts=dict(class_counts),
            avg_confidence=float(np.mean(confs)) if confs else 0.0,
            min_confidence=float(np.min(confs)) if confs else 0.0,
            max_confidence=float(np.max(confs)) if confs else 0.0,
            events=[e.to_dict() for e in events[:500]], track_spans=track_spans,
            processing_time_s=processing_time_s,
            model_identities=model_identities,
            unique_tracks=len(track_persistence),
            track_persistence=dict(track_persistence),
            person_track_stability=person_stability,
            duplicate_detections=duplicate_detections,
            stopped_transitions=planner_events["STOPPED"],
            path_blocked_transitions=planner_events["PATH_BLOCKED"],
            replanning_events=planner_events["REPLANNING"],
            track_switches=track_switches,
            notes=[
                "Same AnalysisPipeline as live camera.",
                "Timestamps approx. from source FPS / frame index.",
                "Distances and velocity remain IMAGE-SPACE; no metric calibration was applied.",
                "This report records observations and does not establish detector superiority.",
            ],
        )
