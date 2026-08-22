from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional

from .detector import Detection
from .geometry import box_iou


@dataclass
class ModelEvidence:
    model: str
    model_type: str
    model_version: str
    weights: str
    status: str
    latency_ms: float
    detections: List[dict] = field(default_factory=list)
    configuration: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrchestrationResult:
    detections: List[Detection]
    evidence: List[ModelEvidence]
    fusion: dict
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "evidence": [e.to_dict() for e in self.evidence],
            "fusion": dict(self.fusion),
            "notes": list(self.notes),
        }


class PerceptionOrchestrator:
    """Run independent perception sources and fuse evidence without inventing labels."""

    def __init__(self, sources: Optional[Mapping[str, object]] = None,
                 fusion_iou_threshold: float = 0.5):
        self.sources: Dict[str, object] = dict(sources or {})
        self.fusion_iou_threshold = float(fusion_iou_threshold)

    def set_source(self, name: str, detector: object) -> None:
        self.sources[str(name)] = detector

    def remove_source(self, name: str) -> None:
        self.sources.pop(str(name), None)

    def infer(self, frame, timestamp: Optional[float] = None,
              frame_id: Optional[int] = None) -> OrchestrationResult:
        evidence: List[ModelEvidence] = []
        raw: List[Detection] = []
        notes: List[str] = []
        for source_name, detector in self.sources.items():
            identity = dict(getattr(detector, "identity", {}) or {})
            model = str(identity.get("model", source_name))
            model_type = str(identity.get("model_type", "UNKNOWN"))
            model_version = str(identity.get("model_version", "UNKNOWN"))
            weights = str(identity.get("weights", "UNKNOWN"))
            started = time.perf_counter()
            try:
                detections = list(detector.detect(frame, timestamp=timestamp, frame_id=frame_id) or [])
                latency_ms = (time.perf_counter() - started) * 1000.0
                raw.extend(detections)
                evidence.append(ModelEvidence(
                    model=model, model_type=model_type, model_version=model_version,
                    weights=weights, status="AVAILABLE", latency_ms=round(latency_ms, 3),
                    detections=[d.to_dict() for d in detections],
                    configuration=self._configuration(detector),
                ))
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                message = f"{type(exc).__name__}: {exc}"
                notes.append(f"{model}: unavailable; evidence source skipped")
                evidence.append(ModelEvidence(
                    model=model, model_type=model_type, model_version=model_version,
                    weights=weights, status="UNAVAILABLE", latency_ms=round(latency_ms, 3),
                    configuration=self._configuration(detector), error=message,
                ))

        fused, merge_count = self._fuse(raw)
        active = [e.model for e in evidence if e.status == "AVAILABLE"]
        fusion = {
            "mode": "MAX_CONFIDENCE_BY_CLASS_AND_IOU",
            "raw_detections": len(raw),
            "fused_detections": len(fused),
            "merged_detections": merge_count,
            "active_sources": active,
            "ground_truth": False,
            "ground_truth_note": "Fusion is an inference-time evidence operation, not annotation.",
        }
        return OrchestrationResult(detections=fused, evidence=evidence, fusion=fusion, notes=notes)

    def _fuse(self, detections: Iterable[Detection]):
        fused: List[Detection] = []
        merge_count = 0
        for detection in sorted(list(detections), key=lambda d: float(d.confidence), reverse=True):
            match_index = None
            for index, existing in enumerate(fused):
                if existing.class_name == detection.class_name and box_iou(existing.bbox, detection.bbox) >= self.fusion_iou_threshold:
                    match_index = index
                    break
            if match_index is None:
                fused.append(detection)
                continue
            existing = fused[match_index]
            sources = tuple(dict.fromkeys(
                list(getattr(existing, "source_models", ()) or (existing.source_model,))
                + list(getattr(detection, "source_models", ()) or (detection.source_model,))
            ))
            best = existing if existing.confidence >= detection.confidence else detection
            fused[match_index] = replace(
                best,
                source_model="PERCEPTION FUSION",
                source_models=sources,
                agreement_count=len(sources),
            )
            merge_count += 1
        return fused, merge_count

    @staticmethod
    def _configuration(detector: object) -> dict:
        config = getattr(detector, "config", None)
        if isinstance(config, dict):
            return dict(config)
        fields = (
            "conf_threshold", "iou_threshold", "device", "image_size",
            "model_path", "classes", "max_detections",
        )
        return {name: getattr(detector, name) for name in fields if hasattr(detector, name)}
