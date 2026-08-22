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
from ..vision.temporal_smoothing import TemporalSmoother
from ..vision.perception_config import (
    DEFAULT_CLASS_MAPPING,
    PERCEPTION_CURRENT,
    PERCEPTION_YOLO_BASELINE,
    PERCEPTION_FUSION,
    PERCEPTION_ARQTECH,
    PerceptionConfig,
)
from ..vision.yolo_adapter import YoloDetector
from ..vision.orchestrator import PerceptionOrchestrator, OrchestrationResult
from ..planning.obstacle_fusion import fuse_obstacles, fusion_stats
from ..brain.risk_engine import RiskEngine, RiskAssessment
from ..brain.decision_engine import DecisionEngine, SceneDecision
from ..brain.uncertainty import UncertaintyEngine, UncertaintyReport
from ..planning.image_planner import ImageSpacePlanner, ImagePlanResult
from ..planning.occupancy import build_occupancy_from_mask, build_cost_map, OccupancyGrid
from ..vision.navigation_relevance import enrich_detections
from ..vision.scene_narrative import build_narrative, scene_inventory
from ..robotics.world_model import WorldModel
from ..robotics.navigation_state import NavigationController
from ..integrations import GroqClient
from ..segmentation import ContourSegmenter
from ..motion import MotionEngine
from ..planning.semantic_occupancy import build_semantic_occupancy
from ..planning.cost_map import build_navigation_cost_map
from ..simulation import RobotSimulation
try:
    from ..arqtech.inference import ArqtechDetector
except Exception:
    ArqtechDetector = None
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
    simulation_overlay: Optional[np.ndarray]
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
    model_identity: dict = field(default_factory=dict)
    requested_perception_mode: str = PERCEPTION_CURRENT
    smoothing: dict = field(default_factory=dict)
    calibration_status: str = "NOT CALIBRATED"
    telemetry: dict = field(default_factory=dict)
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None
    source: str = "unknown"
    perception_evidence: List[dict] = field(default_factory=list)
    perception_fusion: dict = field(default_factory=dict)
    groq_analysis: dict = field(default_factory=dict)
    segmentation_report: dict = field(default_factory=dict)
    motion_observations: List[dict] = field(default_factory=list)
    motion_events: List[dict] = field(default_factory=list)
    trajectory_heatmap: dict = field(default_factory=dict)
    semantic_occupancy: dict = field(default_factory=dict)
    navigation_cost_map: dict = field(default_factory=dict)
    simulation_state: dict = field(default_factory=dict)

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
            "model": self.model_identity.get("model", "UNKNOWN"),
            "model_type": self.model_identity.get("model_type", "UNKNOWN"),
            "model_version": self.model_identity.get("model_version", "UNKNOWN"),
            "weights": self.model_identity.get("weights", "UNKNOWN"),
            "calibration_status": self.calibration_status,
            "smoothing": self.smoothing,
            "telemetry": self.telemetry,
            "track_events": len(self.track_events),
            "perception_evidence": len(self.perception_evidence),
            "perception_fusion": self.perception_fusion,
            "groq": self.groq_analysis,
            "segmentation": self.segmentation_report,
            "motion_observations": len(self.motion_observations),
            "motion_events": len(self.motion_events),
            "trajectory_heatmap": {k: v for k, v in self.trajectory_heatmap.items() if k != "array"},
            "semantic_occupancy": self.semantic_occupancy,
            "navigation_cost_map": self.navigation_cost_map,
            "simulation": self.simulation_state,
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
    def __init__(self, min_area=80, conf_threshold=0.35, cell_size=16,
                 max_image_side=1280, enable_tracking=False,
                 perception_mode=PERCEPTION_CURRENT, model_path="yolo11n.pt",
                 iou_threshold=0.45, device="auto", image_size=640,
                 classes=None, max_detections=100, tracker_type="IOU",
                 smoothing_enabled=False, smoothing_method="RAW",
                 smoothing_window=5, smoothing_alpha=0.35, class_mapping=None,
                 calibration_status="NOT CALIBRATED", enable_groq=False, groq_client=None,
                 arqtech_checkpoint=None, enable_segmentation=True, segmenter=None):
        mapping = dict(DEFAULT_CLASS_MAPPING) if class_mapping is None else dict(class_mapping)
        self.config = PerceptionConfig(
            mode=perception_mode,
            model_path=model_path,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            image_size=image_size,
            classes=classes,
            max_detections=max_detections,
            tracker_type=tracker_type,
            tracking_enabled=enable_tracking,
            smoothing_enabled=smoothing_enabled,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_alpha=smoothing_alpha,
            class_mapping=mapping,
            calibration_status=calibration_status,
        ).normalized()
        self.preprocessor = Preprocessor(max_side=max_image_side)
        self.current_detector = ClassicalDetector(min_area=min_area, conf_threshold=self.config.confidence_threshold)
        self.detector = self.current_detector  # compatibility for callers using the old attribute
        self.yolo_detector = None
        self.arqtech_detector = None
        self.arqtech_status = "NOT CONFIGURED"
        self.active_detector = self.current_detector
        self.model_identity = self.current_detector.identity
        self.requested_model_identity = self.config.model_identity()
        self.detector_status = "AVAILABLE"
        self.detector_error = ""
        if self.config.mode in (PERCEPTION_YOLO_BASELINE, PERCEPTION_FUSION):
            self.yolo_detector = YoloDetector(
                model_path=self.config.model_path,
                conf_threshold=self.config.confidence_threshold,
                iou_threshold=self.config.iou_threshold,
                device=self.config.device,
                image_size=self.config.image_size,
                classes=list(self.config.classes) if self.config.classes is not None else None,
                max_detections=self.config.max_detections,
            )
            if self.yolo_detector.available:
                if self.config.mode == PERCEPTION_YOLO_BASELINE:
                    self.active_detector = self.yolo_detector
                    self.model_identity = self.yolo_detector.identity
                elif self.config.mode == PERCEPTION_FUSION:
                    self.model_identity = self.config.model_identity()
            else:
                self.detector_status = "UNAVAILABLE"
                self.detector_error = self.yolo_detector.error or "Ultralytics/model unavailable"
        if self.config.mode in (PERCEPTION_ARQTECH, PERCEPTION_FUSION) and arqtech_checkpoint and ArqtechDetector is not None:
            self.arqtech_detector = ArqtechDetector(checkpoint_path=arqtech_checkpoint, device=self.config.device)
            self.arqtech_status = "AVAILABLE" if self.arqtech_detector.available else "UNAVAILABLE"
            if self.config.mode == PERCEPTION_ARQTECH and self.arqtech_detector.available:
                self.active_detector = self.arqtech_detector
                self.model_identity = self.arqtech_detector.identity
        elif self.config.mode == PERCEPTION_ARQTECH:
            self.arqtech_status = "UNAVAILABLE"
        orchestrator_sources = {}
        if self.config.mode == PERCEPTION_ARQTECH and self.arqtech_detector is not None and self.arqtech_detector.available:
            orchestrator_sources["ARQTECH"] = self.arqtech_detector
        elif self.config.mode == PERCEPTION_FUSION:
            orchestrator_sources["CURRENT DETECTOR"] = self.current_detector
            if self.yolo_detector is not None and self.yolo_detector.available:
                orchestrator_sources["YOLO"] = self.yolo_detector
            if self.arqtech_detector is not None and self.arqtech_detector.available:
                orchestrator_sources["ARQTECH"] = self.arqtech_detector
        elif self.config.mode == PERCEPTION_YOLO_BASELINE and self.yolo_detector is not None and self.yolo_detector.available:
            orchestrator_sources["YOLO"] = self.yolo_detector
        else:
            orchestrator_sources["CURRENT DETECTOR"] = self.current_detector
        self.orchestrator = PerceptionOrchestrator(orchestrator_sources, fusion_iou_threshold=self.config.iou_threshold)
        self.last_orchestration = OrchestrationResult([], [], {}, [])
        self.scene_analyzer = SceneAnalyzer()
        self.risk_engine = RiskEngine()
        self.uncertainty_engine = UncertaintyEngine()
        self.planner = ImageSpacePlanner(cell_size=cell_size)
        self.brain = DecisionEngine()
        self.tracker = None
        if self.config.tracking_enabled and self.config.tracker_type == "IOU":
            self.tracker = IoUTracker(
                max_age=self.config.tracker_max_age,
                min_hits=self.config.tracker_min_hits,
                iou_threshold=self.config.tracker_iou_threshold,
            )
        self.enable_tracking = self.tracker is not None
        self.smoother = TemporalSmoother(
            enabled=self.config.smoothing_enabled,
            method=self.config.smoothing_method,
            window_size=self.config.smoothing_window,
            alpha=self.config.smoothing_alpha,
        )
        self.cell_size = cell_size
        self.nav_controller = NavigationController()
        self.last_dropped_frames = 0
        self.groq_client = groq_client if groq_client is not None else (GroqClient() if enable_groq else None)
        self.groq_enabled = bool(enable_groq)
        self.segmenter = segmenter if segmenter is not None else (ContourSegmenter() if enable_segmentation else None)
        self.segmentation_enabled = self.segmenter is not None
        self.motion_engine = MotionEngine()
        self.simulation = RobotSimulation()

    def reset_tracker(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.smoother.reset()
        self.motion_engine.reset()
        self.simulation.reset()
        self.nav_controller.reset()

    def _detect(self, work, timestamp, frame_id, notes):
        if self.config.mode in (PERCEPTION_YOLO_BASELINE, PERCEPTION_FUSION) and self.detector_status == "UNAVAILABLE":
            notes.append("YOLO BASELINE: UNAVAILABLE")
            notes.append("Fallback: CURRENT DETECTOR")
        if self.config.mode == PERCEPTION_ARQTECH and self.arqtech_status != "AVAILABLE":
            notes.append("ARQTECH: UNAVAILABLE FOR DETECTION")
            notes.append("Fallback: CURRENT DETECTOR")
        self.last_orchestration = self.orchestrator.infer(work, timestamp=timestamp, frame_id=frame_id)
        notes.extend(self.last_orchestration.notes)
        return list(self.last_orchestration.detections)

    def run_packet(self, packet, run_planner: bool = True) -> AnalysisResult:
        if packet is None or getattr(packet, "frame", None) is None:
            raise ValueError("Empty or invalid FramePacket")
        metadata = getattr(packet, "metadata", {}) or {}
        return self.run(
            packet.frame,
            run_planner=run_planner,
            timestamp=getattr(packet, "timestamp", None),
            frame_id=getattr(packet, "frame_id", None),
            source=getattr(packet, "source", "unknown"),
            source_fps=getattr(packet, "fps", None),
            dropped_frames=metadata.get("dropped_frames"),
        )

    def compare_models(self, image_bgr: np.ndarray, timestamp: Optional[float] = None,
                       frame_id: Optional[int] = None) -> dict:
        """Run the two detector baselines on one frame; no quality claim is inferred."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or invalid image")
        timestamp = time.time() if timestamp is None else float(timestamp)
        work = self.preprocessor.run(image_bgr).resized
        yolo = self.yolo_detector
        if yolo is None:
            yolo = YoloDetector(
                model_path=self.config.model_path,
                conf_threshold=self.config.confidence_threshold,
                iou_threshold=self.config.iou_threshold,
                device=self.config.device,
                image_size=self.config.image_size,
                classes=list(self.config.classes) if self.config.classes is not None else None,
                max_detections=self.config.max_detections,
            )
        rows = {}
        for label, detector, identity in (
            ("CURRENT DETECTOR", self.current_detector, self.current_detector.identity),
            ("YOLO BASELINE", yolo, yolo.identity),
        ):
            started = time.perf_counter()
            error = None
            try:
                detections = detector.detect(work, timestamp=timestamp, frame_id=frame_id)
                detections = nms_detections(detections, iou_threshold=self.config.iou_threshold)
            except Exception as exc:
                detections = []
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000.0
            rows[label] = {
                "model_identity": identity,
                "detections_per_frame": len(detections),
                "average_confidence": round(float(np.mean([d.confidence for d in detections])), 3) if detections else 0.0,
                "inference_latency_ms": round(latency_ms, 3),
                "inference_fps": round(1000.0 / latency_ms, 3) if latency_ms > 0 else "N/A",
                "error": error,
            }
        return {
            "comparison": rows,
            "methodology": "Same preprocessed frame; detector outputs only; no ground truth and no quality conclusion.",
            "calibration_status": self.config.calibration_status,
            "notes": ["YOLO is an external baseline and is not ARQTECH.", "Metrics are measured only for this frame."],
        }

    def run(self, image_bgr: np.ndarray, run_planner: bool = True,
            timestamp: Optional[float] = None, frame_id: Optional[int] = None,
            source: str = "unknown", source_fps: Optional[float] = None,
            dropped_frames: Optional[int] = None) -> AnalysisResult:
        t0 = time.perf_counter()
        timestamp = time.time() if timestamp is None else float(timestamp)
        lat = LatencyBreakdown()
        notes = [
            "Pipeline: Preprocess → Detector → Normalized Detection → Track → Smooth → Fuse → Scene → Occupancy → Planner → Decision.",
            "Navigation path and distance are image-space only.",
            "ARQTECH is an experimental/future architecture; it is not YOLO.",
            "Calibration status: NOT CALIBRATED; metric distance and real-world speed are unavailable.",
        ]
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or invalid image")
        with measure("preprocess", lat):
            prep = self.preprocessor.run(image_bgr)
        work = prep.resized
        h, w = work.shape[:2]
        with measure("detection", lat):
            raw_detections = self._detect(work, timestamp, frame_id, notes)
            detections = nms_detections(raw_detections, iou_threshold=self.config.iou_threshold)
            if len(raw_detections) != len(detections):
                notes.append(f"NMS: {len(raw_detections)} → {len(detections)} detections.")
        segmentation_report = {"status": "DISABLED", "mask_count": 0, "method": "NONE"}
        with measure("segmentation", lat):
            if self.segmenter is not None:
                try:
                    segmentation = self.segmenter.segment(work, detections, timestamp=timestamp, frame_id=frame_id)
                    detections = list(segmentation.detections)
                    segmentation_report = segmentation.to_dict()
                except Exception as exc:
                    notes.append(f"Segmentation unavailable: {type(exc).__name__}")
                    segmentation_report = {
                        "status": "FAILED", "mask_count": 0,
                        "method": getattr(self.segmenter, "model_name", "UNKNOWN"),
                        "notes": ["Original bbox detections preserved."],
                    }
            else:
                notes.append("Segmentation disabled; bbox-only geometry remains active.")
        tracks = []
        tracking_active = False
        track_events = []
        with measure("tracking", lat):
            if self.tracker is not None:
                tracks = self.tracker.update(detections, timestamp=timestamp, frame_id=frame_id)
                self.smoother.update(tracks)
                tracking_active = self.tracker.active
                track_events = list(getattr(self.tracker, "events", []) or [])
            else:
                notes.append("Tracking disabled (static image mode).")
        motion_observations = []
        motion_events = []
        trajectory_heatmap = {"status": "EMPTY", "sample_count": 0, "track_count": 0, "unit": "IMAGE-SPACE PROJECTION", "physical_map": False}
        with measure("motion", lat):
            if tracks:
                motion_observations_raw, motion_events = self.motion_engine.update(tracks, timestamp=timestamp)
                motion_observations = [observation.to_dict() for observation in motion_observations_raw]
                trajectory_heatmap = self.motion_engine.heatmap((h, w))
        fused = fuse_obstacles(
            detections,
            tracks=tracks if tracks else None,
            min_relevance="MEDIUM",
            class_mapping=self.config.class_mapping,
        )
        fstats = fusion_stats(len(detections), fused)
        with measure("scene", lat):
            scene = self.scene_analyzer.analyze(work, detections)
        avg_conf = float(np.mean([d.confidence for d in detections])) if detections else 0.5
        occupancy = None
        cost_map = None
        semantic_occupancy = None
        navigation_cost_map = None
        with measure("occupancy", lat):
            if scene.free_space_mask is not None:
                occupancy = build_occupancy_from_mask(scene.free_space_mask, cell_size=self.cell_size)
                semantic_occupancy = build_semantic_occupancy(
                    (h, w), free_space_mask=scene.free_space_mask,
                    detections=detections, tracks=tracks, cell_size=self.cell_size,
                )
        with measure("risk", lat):
            risk = self.risk_engine.assess(
                scene, path_available=True, avg_confidence=avg_conf,
                detections=detections, motion_observations=motion_observations,
                image_shape=(h, w),
            )
            if occupancy is not None:
                navigation_cost_map = build_navigation_cost_map(
                    occupancy, risk_assessment=risk,
                    predicted_trajectories=motion_observations,
                )
                cost_map = navigation_cost_map.grid
        plan = None
        comparison = []
        with measure("planning", lat):
            if run_planner and scene.free_space_mask is not None:
                comparison = self.planner.compare(
                    scene.free_space_mask, cost_map=cost_map,
                )
                plan = next((p for p in comparison if p.algorithm == "astar"), comparison[0] if comparison else None)
                risk = self.risk_engine.assess(
                    scene, path_available=plan.success if plan else False,
                    avg_confidence=avg_conf, detections=detections,
                    motion_observations=motion_observations, image_shape=(h, w),
                )
                if occupancy is not None:
                    navigation_cost_map = build_navigation_cost_map(
                        occupancy, risk_assessment=risk,
                        predicted_trajectories=motion_observations,
                    )
                    cost_map = navigation_cost_map.grid
        with measure("decision", lat):
            decision = self.brain.decide(
                scene.estimated_free_space_ratio,
                scene.obstacle_density,
                risk.level,
                plan.success if plan else False,
                scene.person_count,
            )
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
        groq_analysis = {}
        if self.groq_client is not None:
            with measure("groq", lat):
                groq_result = self.groq_client.analyze_image(
                    work,
                    prompt=(
                        "Analyze this frame as an external multimodal assistant. Return JSON with "
                        "scene_summary, ambiguous_objects, possible_classes and review_questions. "
                        "Do not claim ground truth, metric distance, velocity, control authority or training labels."
                    ),
                    json_mode=True,
                )
            groq_analysis = groq_result.to_dict()
        with measure("render", lat):
            annotated = annotate_detections(work, detections)
            free_overlay = overlay_free_space(work, scene.free_space_mask) if scene.free_space_mask is not None else work.copy()
            path_img = free_overlay.copy()
            if plan and plan.path_px:
                path_img = draw_path(path_img, plan.path_px)
        elapsed = (time.perf_counter() - t0) * 1000.0
        enriched = enrich_detections(detections, w, h, class_mapping=self.config.class_mapping)
        geometry_by_id = {int(g.detection_id): g.to_dict() for g in geometries if g.detection_id is not None}
        motion_by_id = {int(m.get("track_id")): m for m in motion_observations if m.get("track_id") is not None}
        risk_by_id = {int(r.get("object_id")): r for r in risk.object_risks if r.get("object_id") is not None}
        for row, detection in zip(enriched, detections):
            object_id = detection.object_id
            row["geometry"] = geometry_by_id.get(int(object_id), {}) if object_id is not None else {}
            row["motion"] = motion_by_id.get(int(object_id), {}) if object_id is not None else {}
            row["risk"] = risk_by_id.get(int(object_id), {}) if object_id is not None else {}
            row["mask_available"] = detection.mask is not None
        path_ok = bool(plan and plan.success)
        narrative = build_narrative(detections, scene.estimated_free_space_ratio, path_ok, decision.action, risk.level, enriched)
        inv = scene_inventory(detections)
        alternative = next((p for p in comparison if plan is not None and p.algorithm != plan.algorithm), None)
        nav_dict = self.nav_controller.update(
            has_path=path_ok,
            free_space_ratio=float(scene.estimated_free_space_ratio),
            obstacle_density=float(scene.obstacle_density),
            risk_level=risk.level,
            path_length=float(getattr(plan, "path_length_px", 0) or 0) if plan else 0.0,
            nodes=int(getattr(plan, "nodes_explored", 0) or 0) if plan else 0,
        )
        planner_diag = nav_dict.get("diagnostics")
        simulation_state = self.simulation.update(
            (h, w),
            current_path=plan.path_px if plan is not None else [],
            alternative_path=alternative.path_px if alternative is not None else [],
            navigation_state=nav_dict.get("status"),
            obstacles=fused,
            risk_zones=risk.risk_zones,
        )
        simulation_overlay = self.simulation.render(path_img, simulation_state)
        wm = WorldModel.from_enriched(
            enriched, scene.estimated_free_space_ratio, path_ok,
            frame_id=frame_id or 0,
            semantic_occupancy=semantic_occupancy.to_dict() if semantic_occupancy is not None else {},
            trajectories=self.motion_engine.trajectories(), risk_zones=risk.risk_zones,
            occupancy=occupancy, current_path=plan, alternative_path=alternative,
            simulation=simulation_state.to_dict(),
        )
        detection_ms = float(lat.stages.get("detection", 0.0))
        telemetry = {
            "source_fps": round(float(source_fps), 3) if source_fps is not None else "N/A",
            "inference_fps": round(1000.0 / detection_ms, 3) if detection_ms > 0 else "N/A",
            "pipeline_fps": round(1000.0 / elapsed, 3) if elapsed > 0 else "N/A",
            "inference_latency_ms": round(detection_ms, 3) if detection_ms > 0 else "N/A",
            "tracking_latency_ms": round(float(lat.stages.get("tracking", 0.0)), 3),
            "segmentation_latency_ms": round(float(lat.stages.get("segmentation", 0.0)), 3),
            "motion_latency_ms": round(float(lat.stages.get("motion", 0.0)), 3),
            "planning_latency_ms": round(float(lat.stages.get("planning", 0.0)), 3),
            "total_latency_ms": round(elapsed, 3),
            "dropped_frames": int(dropped_frames) if dropped_frames is not None else "N/A",
        }
        if self.tracker is not None:
            telemetry["track_switches"] = self.tracker.track_switches
        actual_identity = dict(self.model_identity)
        if self.config.mode == PERCEPTION_YOLO_BASELINE and self.detector_status == "UNAVAILABLE":
            actual_identity["requested_model"] = "YOLO"
            actual_identity["fallback"] = "CURRENT DETECTOR"
            actual_identity["fallback_reason"] = self.detector_error
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
            fusion_stats=fstats, track_events=track_events,
            planner_diagnostics=planner_diag, model_identity=actual_identity,
            requested_perception_mode=self.config.mode,
            smoothing=self.smoother.stats(),
            calibration_status=self.config.calibration_status,
            telemetry=telemetry, timestamp=timestamp, frame_id=frame_id, source=source,
            perception_evidence=[e.to_dict() for e in self.last_orchestration.evidence],
            perception_fusion=dict(self.last_orchestration.fusion),
            groq_analysis=groq_analysis,
            segmentation_report=segmentation_report,
            motion_observations=motion_observations,
            motion_events=motion_events,
            trajectory_heatmap=trajectory_heatmap,
            semantic_occupancy=semantic_occupancy.to_dict() if semantic_occupancy is not None else {},
            navigation_cost_map=navigation_cost_map.to_dict() if navigation_cost_map is not None else {},
            simulation_state=simulation_state.to_dict(),
            simulation_overlay=simulation_overlay,
        )
