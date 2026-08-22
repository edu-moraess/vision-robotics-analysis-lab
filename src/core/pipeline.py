"""End-to-end analysis pipeline with measured latency breakdown."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..vision.detector import ClassicalDetector, Detection
from ..vision.scene import SceneAnalyzer, SceneAnalysis
from ..vision.annotator import annotate_detections, overlay_free_space, draw_path
from ..vision.geometry import GeometryEngine, ObjectGeometry
from ..vision.preprocessing import Preprocessor, PreprocessResult
from ..vision.tracker import IoUTracker, Track
from ..vision.nms import nms_detections
from ..planning.obstacle_fusion import fuse_obstacles, fusion_stats
from ..brain.risk_engine import RiskEngine, RiskAssessment
from ..brain.decision_engine import DecisionEngine, SceneDecision
from ..brain.uncertainty import UncertaintyEngine, UncertaintyReport
from ..planning.image_planner import ImageSpacePlanner, ImagePlanResult
from ..planning.occupancy import build_occupancy_from_mask, build_cost_map, OccupancyGrid
from ..vision.navigation_relevance import enrich_detections
from ..vision.scene_narrative import build_narrative, scene_inventory
from ..robotics.world_model import WorldModel
from ..robotics.navigation_state import derive_navigation_state, NavigationController
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
    geometries: List[ObjectGeometry] = field(default_factory=list)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    notes: List[str] = field(default_factory=list)
    narrative: List[str] = field(default_factory=list)
    inventory: dict = field(default_factory=dict)
    enriched_detections: List[dict] = field(default_factory=list)
    world_model: Optional[dict] = None
    navigation_state: Optional[dict] = None
    fused_obstacles: List[dict] = field(default_factory=list)
    fusion_stats: Optional[dict] = None
    track_events: List[dict] = field(default_factory=list)
    planner_diagnostics: Optional[dict] = None

    def metrics(self) -> Dict[str, Any]:
        avg_conf = float(np.mean([d.confidence for d in self.detections])) if self.detections else 0.0
        out = {
            "processing_time_ms": round(self.processing_time_ms, 2),
            "detection_count": len(self.detections),
            "fused_obstacle_count": len(self.fused_obstacles),
            "nav_status": (self.navigation_state or {}).get("status"),
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
        if self.fusion_stats:
            out["fusion"] = self.fusion_stats
        if self.planner_diagnostics:
            out["planner_diagnostics"] = self.planner_diagnostics
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
        self.nav_controller = NavigationController()

    def reset_tracker(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.nav_controller.reset()

    def run(self, image_bgr: np.ndarray, run_planner: bool = True) -> AnalysisResult:
        t0 = time.perf_counter()
        lat = LatencyBreakdown()
        notes = [
            "Pipeline: Preprocess → ClassicalDetector → NMS → Track → Fuse → Scene → Occupancy → Planner → Decision.",
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
            raw_detections = self.detector.detect(work)
            detections = nms_detections(raw_detections, iou_threshold=0.45)
            if len(raw_detections) != len(detections):
                notes.append(f"NMS: {len(raw_detections)} → {len(detections)} detections.")
        tracks = []
        tracking_active = False
        track_events = []
        with measure("tracking", lat):
            if self.tracker is not None:
                tracks = self.tracker.update(detections)
                tracking_active = self.tracker.active
                track_events = list(getattr(self.tracker, "events", []) or [])
            else:
                notes.append("Tracking disabled (static image mode).")
        fused = fuse_obstacles(detections, tracks=tracks if tracks else None, min_relevance="MEDIUM")
        fstats = fusion_stats(len(detections), fused)
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
        with measure("geometry", lat):
            geometries = GeometryEngine().analyze(detections, (h, w))
        with measure("render", lat):
            annotated = annotate_detections(work, detections)
            free_overlay = overlay_free_space(work, scene.free_space_mask) if scene.free_space_mask is not None else work.copy()
            path_img = free_overlay.copy()
            if plan and plan.path_px:
                path_img = draw_path(path_img, plan.path_px)
        elapsed = (time.perf_counter() - t0) * 1000.0
        enriched = enrich_detections(detections, w, h)
        path_ok = bool(plan and plan.success)
        narrative = build_narrative(detections, scene.estimated_free_space_ratio, path_ok, decision.action, risk.level, enriched)
        inv = scene_inventory(detections)
        wm = WorldModel.from_enriched(enriched, scene.estimated_free_space_ratio, path_ok)
        nav_dict = self.nav_controller.update(
            has_path=path_ok,
            free_space_ratio=float(scene.estimated_free_space_ratio),
            obstacle_density=float(scene.obstacle_density),
            risk_level=risk.level,
            path_length=float(getattr(plan, "path_length_px", 0) or 0) if plan else 0.0,
            nodes=int(getattr(plan, "nodes_explored", 0) or 0) if plan else 0,
        )
        planner_diag = nav_dict.get("diagnostics")
        return AnalysisResult(
            detections=detections, scene=scene, risk=risk, plan=plan,
            plan_comparison=comparison, decision=decision,
            annotated_image=annotated, free_space_overlay=free_overlay,
            path_overlay=path_img, processing_time_ms=elapsed, image_shape=(h, w),
            preprocess=prep, occupancy=occupancy, cost_map=cost_map,
            tracks=tracks, tracking_active=tracking_active,
            uncertainty=unc, geometries=geometries, latency=lat, notes=notes,
            narrative=narrative, inventory=inv, enriched_detections=enriched,
            world_model=wm.to_dict(), navigation_state=nav_dict,
            fused_obstacles=[o.to_dict() for o in fused],
            fusion_stats=fstats,
            track_events=track_events,
            planner_diagnostics=planner_diag,
        )
