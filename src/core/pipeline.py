"""End-to-end pipeline: Detection → Scene → Risk → Planner → Decision."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..vision.detector import ClassicalDetector, Detection
from ..vision.scene import SceneAnalyzer, SceneAnalysis
from ..vision.annotator import annotate_detections, overlay_free_space, draw_path
from ..vision.geometry import resize_keep_aspect
from ..brain.risk_engine import RiskEngine, RiskAssessment
from ..brain.decision_engine import DecisionEngine, SceneDecision
from ..planning.image_planner import ImageSpacePlanner, ImagePlanResult

@dataclass
class AnalysisResult:
    detections: List[Detection]
    scene: SceneAnalysis
    risk: RiskAssessment
    plan: Optional[ImagePlanResult]
    plan_comparison: List[ImagePlanResult]
    decision: SceneDecision
    annotated_image: np.ndarray
    free_space_overlay: np.ndarray
    path_overlay: np.ndarray
    processing_time_ms: float
    image_shape: Tuple[int, int]
    notes: List[str] = field(default_factory=list)

    def metrics(self):
        avg = float(np.mean([d.confidence for d in self.detections])) if self.detections else 0.0
        return {"processing_time_ms": round(self.processing_time_ms, 2),
                "detection_count": len(self.detections), "average_confidence": round(avg, 3),
                "obstacle_density": round(self.scene.obstacle_density, 4),
                "free_space_ratio": round(self.scene.estimated_free_space_ratio, 4),
                "risk_score": round(self.risk.score, 3), "risk_level": self.risk.level,
                "path_success": self.plan.success if self.plan else False,
                "path_length_px": round(self.plan.path_length_px, 1) if self.plan else None,
                "decision": self.decision.action}

class AnalysisPipeline:
    def __init__(self, min_area=80, conf_threshold=0.35, cell_size=16, max_image_side=1280):
        self.detector = ClassicalDetector(min_area=min_area, conf_threshold=conf_threshold)
        self.scene_analyzer = SceneAnalyzer()
        self.risk_engine = RiskEngine()
        self.planner = ImageSpacePlanner(cell_size=cell_size)
        self.brain = DecisionEngine()
        self.max_image_side = max_image_side

    def run(self, image_bgr, run_planner=True):
        t0 = time.perf_counter()
        notes = ["Pipeline: ClassicalDetector → Scene → Risk → Planner → Decision.",
                 "Navigation path is hypothetical and image-space only."]
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or invalid image")
        work, _ = resize_keep_aspect(image_bgr, self.max_image_side)
        h, w = work.shape[:2]
        detections = self.detector.detect(work)
        scene = self.scene_analyzer.analyze(work, detections)
        avg = float(np.mean([d.confidence for d in detections])) if detections else 0.5
        risk = self.risk_engine.assess(scene, True, avg)
        plan, comparison = None, []
        if run_planner and scene.free_space_mask is not None:
            comparison = self.planner.compare(scene.free_space_mask)
            plan = next((p for p in comparison if p.algorithm == "astar"), comparison[0] if comparison else None)
            risk = self.risk_engine.assess(scene, plan.success if plan else False, avg)
        decision = self.brain.decide(scene.estimated_free_space_ratio, scene.obstacle_density,
                                     risk.level, plan.success if plan else False, scene.person_count)
        annotated = annotate_detections(work, detections)
        free_overlay = overlay_free_space(work, scene.free_space_mask) if scene.free_space_mask is not None else work.copy()
        path_img = draw_path(free_overlay.copy(), plan.path_px) if plan and plan.path_px else free_overlay.copy()
        return AnalysisResult(detections, scene, risk, plan, comparison, decision, annotated,
                              free_overlay, path_img, (time.perf_counter()-t0)*1000, (h, w), notes)
