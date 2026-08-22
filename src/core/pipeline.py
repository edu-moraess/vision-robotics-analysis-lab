"""End-to-end analysis pipeline with measured latency breakdown."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..vision.detector import ClassicalDetector, Detection
from ..vision.scene import SceneAnalyzer, SceneAnalysis
from ..vision.annotator import annotate_detections, overlay_free_space, draw_path
from ..vision.preprocessing import Preprocessor, PreprocessResult
from ..vision.tracker import IoUTracker, Track
from ..brain.risk_engine import RiskEngine, RiskAssessment
from ..brain.decision_engine import DecisionEngine, SceneDecision
from ..brain.uncertainty import UncertaintyEngine, UncertaintyReport
from ..planning.image_planner import ImageSpacePlanner, ImagePlanResult
from ..planning.occupancy import build_occupancy_from_mask, build_cost_map, OccupancyGrid
from .timing import LatencyBreakdown, measure

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
    preprocess: Optional[PreprocessResult] = None
    occupancy: Optional[OccupancyGrid] = None
    cost_map: Optional[np.ndarray] = None
    tracks: List[Track] = field(default_factory=list)
    tracking_active: bool = False
    uncertainty: Optional[UncertaintyReport] = None
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    notes: List[str] = field(default_factory=list)

    def metrics(self) -> Dict[str, Any]:
        avg_conf = float(np.mean([d.confidence for d in self.detections])) if self.detections else 0.0
        out = {
            "processing_time_ms": round(self.processing_time_ms, 2),
            "detection_count": len(self.detections),
            "average_confidence": round(avg_conf, 3),
            "obstacle_density": round(self.scene.obstacle_density, 4),
            "free_space_ratio": round(self.scene.estimated_free_space_ratio, 4),
            "risk_score": round(self.risk.score, 3),
            "risk_level": self.risk.level,
            "path_success": self.plan.success if self.plan else False,
            "path_length_px": round(self.plan.path_length_px, 1) if self.plan else None,
            "decision": self.decision.action,
            "tracking_active": self.tracking_active,
            "track_count": len(self.tracks),
        }
        if self.uncertainty is not None:
            out["uncertainty_overall"] = round(self.uncertainty.overall, 3)
        if self.latency is not None and hasattr(self.latency, "to_dict"):
            out["latency_breakdown_ms"] = self.latency.to_dict()
        return out

class AnalysisPipeline:
    def __init__(self, min_area=80, conf_threshold=0.35, cell_size=16, max_image_side=1280, enable_tracking=False):
        self.preprocessor = Preprocessor(max_side=max_image_side)
        self.detector = ClassicalDetector(min_area=min_area, conf_threshold=conf_threshold)
        self.scene_analyzer = SceneAnalyzer()
        self.risk_engine = RiskEngine()
        self.uncertainty_engine = UncertaintyEngine()
        self.planner = ImageSpacePlanner(cell_size=cell_size)
        self.brain = DecisionEngine()
        self.tracker = IoUTracker() if enable_tracking else None
        self.enable_tracking = enable_tracking
        self.cell_size = cell_size

    def reset_tracker(self):
        if self.tracker is not None:
            self.tracker.reset()

    def run(self, image_bgr: np.ndarray, run_planner: bool = True) -> AnalysisResult:
        t0 = time.perf_counter()
        lat = LatencyBreakdown()
        notes = [
            "Pipeline: Preprocess → ClassicalDetector → Scene → Occupancy → Risk → Planner → Decision.",
            "Navigation path is hypothetical and image-space only.",
            "Depth / YOLO / ROS 2 / physical control: not implemented.",
        ]
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or invalid image")
        with measure("preprocess", lat):
            prep = self.preprocessor.run(image_bgr)
        work = prep.resized
        h, w = work.shape[:2]
        with measure("detection", lat):
            detections = self.detector.detect(work)
        tracks = []
        tracking_active = False
        with measure("tracking", lat):
            if self.tracker is not None:
                tracks = self.tracker.update(detections)
                tracking_active = self.tracker.active
            else:
                notes.append("Tracking disabled (static image mode).")
        with measure("scene", lat):
            scene = self.scene_analyzer.analyze(work, detections)
        avg_conf = float(np.mean([d.confidence for d in detections])) if detections else 0.5
        occupancy = None
        cost_map = None
        with measure("occupancy", lat):
            if scene.free_space_mask is not None:
                occupancy = build_occupancy_from_mask(scene.free_space_mask, cell_size=self.cell_size)
                cost_map = build_cost_map(occupancy, inflation=1)
        with measure("risk", lat):
            risk = self.risk_engine.assess(scene, path_available=True, avg_confidence=avg_conf)
        plan = None
        comparison = []
        with measure("planning", lat):
            if run_planner and scene.free_space_mask is not None:
                comparison = self.planner.compare(scene.free_space_mask)
                plan = next((p for p in comparison if p.algorithm == "astar"), comparison[0] if comparison else None)
                risk = self.risk_engine.assess(scene, path_available=plan.success if plan else False, avg_confidence=avg_conf)
        with measure("decision", lat):
            decision = self.brain.decide(scene.estimated_free_space_ratio, scene.obstacle_density, risk.level, plan.success if plan else False, scene.person_count)
        with measure("uncertainty", lat):
            unc = self.uncertainty_engine.assess(
                detections=detections,
                free_space_ratio=scene.estimated_free_space_ratio,
                obstacle_density=scene.obstacle_density,
                path_success=plan.success if plan else False,
                path_nodes=plan.nodes_explored if plan else 0,
                decision_confidence=decision.confidence,
            )
        with measure("render", lat):
            annotated = annotate_detections(work, detections)
            free_overlay = overlay_free_space(work, scene.free_space_mask) if scene.free_space_mask is not None else work.copy()
            path_img = free_overlay.copy()
            if plan and plan.path_px:
                path_img = draw_path(path_img, plan.path_px)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return AnalysisResult(
            detections=detections, scene=scene, risk=risk, plan=plan,
            plan_comparison=comparison, decision=decision,
            annotated_image=annotated, free_space_overlay=free_overlay,
            path_overlay=path_img, processing_time_ms=elapsed, image_shape=(h, w),
            preprocess=prep, occupancy=occupancy, cost_map=cost_map,
            tracks=tracks, tracking_active=tracking_active,
            uncertainty=unc, latency=lat, notes=notes,
        )
