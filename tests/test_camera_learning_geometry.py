from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.camera import VideoFileSource, SmartphoneCameraSource
from src.vision.geometry import GeometryEngine
from src.vision.detector import Detection
from src.learning import ExperienceMemory, FrameCache
from src.core.pipeline import AnalysisPipeline

def test_video_file_missing():
    assert VideoFileSource("/tmp/does_not_exist_vral.mp4").start().online is False

def test_smartphone_empty_url():
    assert SmartphoneCameraSource("").start().online is False

def test_geometry_engine_regions():
    dets = [Detection("obstacle", 0.9, (10, 10, 30, 40), (20, 25)),
            Detection("person", 0.8, (200, 10, 240, 50), (220, 30))]
    geos = GeometryEngine().analyze(dets, (100, 300))
    assert len(geos) == 2 and geos[0].area_px2 > 0 and geos[0].to_dict()["unit"] == "pixel"

def test_frame_cache():
    c = FrameCache(max_frames=3)
    for i in range(5):
        c.push(np.zeros((10, 10, 3), dtype=np.uint8), frame_id=i)
    assert len(c) == 3

def test_experience_memory():
    mem = ExperienceMemory(root=str(Path(tempfile.mkdtemp())))
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    s = mem.store(img, "test", [], 0.5, 0.2, "LOW", "FORWARD", 0.4)
    assert s is not None and mem.count() >= 1 and mem.set_review_status(s.sample_id, "accepted")

def test_pipeline_geometry_field():
    r = AnalysisPipeline().run(np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8))
    assert hasattr(r, "geometries")
