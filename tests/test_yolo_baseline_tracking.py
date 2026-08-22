import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.pipeline import AnalysisPipeline
from src.vision.detector import Detection
from src.vision.perception_config import PERCEPTION_YOLO_BASELINE
from src.vision.temporal_smoothing import TemporalSmoother
from src.vision.tracker import IoUTracker, STATUS_CONFIRMED
from src.vision.yolo_adapter import YoloDetector


def test_normalized_detection_contract_contains_provenance():
    det = Detection("person", 0.82, (1, 2, 11, 22), (6, 12))
    data = det.to_dict()
    assert data["class_id"] is None
    assert data["source_model"] == "CURRENT DETECTOR"
    assert data["model_type"] == "EXISTING CLASSICAL DETECTOR"
    assert data["frame_id"] is None
    assert data["cx"] == 6.0 and data["cy"] == 12.0


def test_yolo_unavailable_falls_back_without_crashing(monkeypatch):
    fake = types.ModuleType("ultralytics")

    class BrokenYOLO:
        def __init__(self, _path):
            raise FileNotFoundError("weights missing")

    fake.YOLO = BrokenYOLO
    fake.__version__ = "test"
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    detector = YoloDetector("missing.pt")
    assert detector.available is False
    pipe = AnalysisPipeline(perception_mode=PERCEPTION_YOLO_BASELINE)
    result = pipe.run(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result.model_identity["model"] == "CURRENT DETECTOR"
    assert any("YOLO BASELINE: UNAVAILABLE" in note for note in result.notes)


def test_yolo_adapter_normalizes_ultralytics_result(monkeypatch):
    fake = types.ModuleType("ultralytics")

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
        def cpu(self):
            return self
        def numpy(self):
            return self.value

    class Boxes:
        xyxy = Tensor([[1, 2, 21, 42]])
        conf = Tensor([0.91])
        cls = Tensor([0])

    class Result:
        names = {0: "person"}
        boxes = Boxes()

    class YOLO:
        names = {0: "person"}
        def __init__(self, _path):
            pass
        def predict(self, **_kwargs):
            return [Result()]

    fake.YOLO = YOLO
    fake.__version__ = "8.test"
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    detector = YoloDetector("weights.pt")
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), timestamp=3.0, frame_id=7)
    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].source_model == "YOLO"
    assert detections[0].model_version == "8.test"
    assert detections[0].frame_id == 7


def test_tracker_ids_and_temporal_history():
    tracker = IoUTracker(min_hits=1, iou_threshold=0.2)
    d1 = Detection("person", 0.8, (10, 10, 30, 30), (20, 20), frame_id=1)
    tracks = tracker.update([d1], timestamp=1.0, frame_id=1)
    assert tracks[0].track_id == d1.object_id
    tracker.update([Detection("person", 0.85, (12, 12, 32, 32), (22, 22), frame_id=2)], timestamp=2.0, frame_id=2)
    track = tracker.confirmed_tracks()[0]
    assert track.status == STATUS_CONFIRMED
    assert len(track.position_history) >= 1
    assert track.position_history[-1]["velocity_unit"] == "IMAGE-SPACE VELOCITY"
    assert any(event["event_type"] in ("OBJECT_CONFIRMED", "OBJECT_UPDATED") for event in tracker.events)


def test_smoothing_keeps_raw_and_smoothed_centers():
    tracker = IoUTracker(min_hits=1, iou_threshold=0.1)
    smoother = TemporalSmoother(enabled=True, method="MOVING_AVERAGE", window_size=2)
    tracker.update([Detection("obstacle", 0.8, (0, 0, 10, 10), (5, 5))])
    tracks = tracker.update([Detection("obstacle", 0.8, (4, 0, 14, 10), (9, 5))])
    smoother.update(tracks)
    track = tracks[0]
    assert track.raw_center == (9.0, 5.0)
    assert track.smoothed_center is not None
    assert track.to_dict()["velocity_unit"] == "IMAGE-SPACE VELOCITY"


def test_baseline_comparison_is_measured_without_quality_claim():
    pipe = AnalysisPipeline()
    report = pipe.compare_models(np.zeros((64, 64, 3), dtype=np.uint8))
    assert set(report["comparison"]) == {"CURRENT DETECTOR", "YOLO BASELINE"}
    assert report["methodology"].startswith("Same preprocessed frame")
    assert "inference_latency_ms" in report["comparison"]["CURRENT DETECTOR"]


def test_orchestrator_fuses_same_class_and_preserves_sources():
    from src.vision.orchestrator import PerceptionOrchestrator

    class StubDetector:
        identity = {
            "model": "STUB",
            "model_type": "TEST SOURCE",
            "model_version": "1",
            "weights": "none",
        }
        def detect(self, frame, timestamp=None, frame_id=None):
            return [Detection("person", 0.9, (10, 10, 30, 30), (20, 20), source_model="STUB")]

    class StubDetector2(StubDetector):
        identity = {
            "model": "STUB2",
            "model_type": "TEST SOURCE",
            "model_version": "1",
            "weights": "none",
        }
        def detect(self, frame, timestamp=None, frame_id=None):
            return [Detection("person", 0.8, (11, 11, 31, 31), (21, 21), source_model="STUB2")]

    result = PerceptionOrchestrator({"a": StubDetector(), "b": StubDetector2()}).infer(np.zeros((40, 40, 3), dtype=np.uint8))
    assert len(result.detections) == 1
    assert result.detections[0].agreement_count == 2
    assert set(result.detections[0].source_models) == {"STUB", "STUB2"}
    assert result.fusion["merged_detections"] == 1


def test_orchestrator_isolates_failed_source():
    from src.vision.orchestrator import PerceptionOrchestrator

    class Broken:
        identity = {"model": "BROKEN", "model_type": "TEST", "model_version": "1", "weights": "none"}
        def detect(self, frame, timestamp=None, frame_id=None):
            raise RuntimeError("boom")

    result = PerceptionOrchestrator({"broken": Broken()}).infer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.detections == []
    assert result.evidence[0].status == "UNAVAILABLE"
    assert result.evidence[0].error
