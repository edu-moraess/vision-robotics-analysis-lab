import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.pipeline import AnalysisPipeline
from src.segmentation import ContourSegmenter
from src.vision.detector import Detection
from src.vision.geometry import GeometryEngine


def _sample_detection():
    return Detection(
        class_name="obstacle", confidence=0.9,
        bbox=(10, 10, 54, 54), center=(32.0, 32.0),
    )


def test_contour_segmenter_produces_estimated_mask_and_perimeter():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (46, 46), (255, 255, 255), thickness=-1)
    result = ContourSegmenter().segment(image, [_sample_detection()])
    assert result.status == "ESTIMATED"
    assert result.mask_count == 1
    detection = result.detections[0]
    assert detection.mask is not None
    assert detection.mask_area_px2 is not None and detection.mask_area_px2 > 0
    assert detection.mask_perimeter_px is not None and detection.mask_perimeter_px > 0
    assert detection.segmentation_status == "ESTIMATED"


def test_geometry_prefers_mask_contour_metrics():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (46, 46), (255, 255, 255), thickness=-1)
    detection = ContourSegmenter().segment(image, [_sample_detection()]).detections[0]
    geometry = GeometryEngine().analyze([detection], image.shape)[0]
    assert geometry.geometry_source == "MASK_CONTOUR"
    assert geometry.mask_available is True
    assert geometry.perimeter_px > 0


def test_pipeline_exposes_segmentation_report():
    result = AnalysisPipeline(enable_segmentation=True).run(
        np.zeros((64, 64, 3), dtype=np.uint8), run_planner=False,
    )
    assert result.segmentation_report["status"] in ("ESTIMATED", "DISABLED")
    assert "segmentation_latency_ms" in result.telemetry
