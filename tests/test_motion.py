import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.motion import MotionEngine, STATIC, MOVING, APPROACHING, CROSSING
from src.vision.detector import Detection
from src.vision.tracker import IoUTracker


def _detections(x, y):
    return [Detection("person", 0.9, (x, y, x + 20, y + 20), (x + 10, y + 10))]


def test_motion_engine_uses_track_history_and_constant_velocity_prediction():
    tracker = IoUTracker(min_hits=1, iou_threshold=0.1)
    motion = MotionEngine(static_speed_threshold=0.1)
    tracks = tracker.update(_detections(10, 10), timestamp=0.0, frame_id=1)
    observations, events = motion.update(tracks, timestamp=0.0)
    assert observations[0].motion_state == STATIC
    tracks = tracker.update(_detections(12, 15), timestamp=1.0, frame_id=2)
    observations, events = motion.update(tracks, timestamp=1.0)
    assert observations[0].motion_state in (MOVING, APPROACHING, CROSSING)
    assert observations[0].predicted_trajectory
    assert observations[0].unit == "IMAGE-SPACE"
    assert any(point["model"] == "CONSTANT VELOCITY" for point in tracks[0].predicted_trajectory)


def test_motion_state_change_is_event_based():
    tracker = IoUTracker(min_hits=1, iou_threshold=0.1)
    motion = MotionEngine(static_speed_threshold=0.1)
    motion.update(tracker.update(_detections(10, 10), timestamp=0.0, frame_id=1), timestamp=0.0)
    observations, events = motion.update(
        tracker.update(_detections(10, 14), timestamp=1.0, frame_id=2), timestamp=1.0,
    )
    assert events and events[0]["event_type"] == "MOTION_STATE_CHANGED"
    assert events[0]["unit"] == "IMAGE-SPACE"


def test_pipeline_exposes_motion_and_heatmap_fields():
    from src.core.pipeline import AnalysisPipeline
    pipeline = AnalysisPipeline(enable_tracking=True)
    first = pipeline.run(np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0, frame_id=1)
    second = pipeline.run(np.zeros((64, 64, 3), dtype=np.uint8), timestamp=1.0, frame_id=2)
    assert "motion_latency_ms" in second.telemetry
    assert isinstance(second.motion_observations, list)
    assert second.trajectory_heatmap["physical_map"] is False
