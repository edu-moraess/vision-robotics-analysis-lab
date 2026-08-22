"""Fuse detections/tracks into navigation obstacles (image-space)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

@dataclass
class NavObstacle:
    obstacle_id: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area_px: float
    relevance: str
    source_class: str
    confidence: float
    track_id: int | None = None

    def to_dict(self):
        return {
            "obstacle_id": self.obstacle_id,
            "bbox": self.bbox,
            "center": self.center,
            "area_px": self.area_px,
            "relevance": self.relevance,
            "source_class": self.source_class,
            "confidence": round(self.confidence, 3),
            "track_id": self.track_id,
        }

RELEVANCE_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

def _relevance_for_class(class_name: str, conf: float, class_mapping=None) -> str:
    name = (class_name or "").lower().strip()
    if class_mapping is not None:
        # An explicit mapping is authoritative: unknown COCO classes are not obstacles.
        return str(class_mapping.get(name, "NONE")).upper()
    if name in ("person", "dynamic", "vehicle", "car"):
        return "HIGH" if conf >= 0.5 else "MEDIUM"
    if name in ("obstacle", "wall", "barrier", "furniture"):
        return "CRITICAL" if conf >= 0.55 else "HIGH"
    if conf < 0.45:
        return "LOW"
    return "MEDIUM"

def fuse_obstacles(detections, tracks=None, min_relevance="MEDIUM", merge_iou=0.3,
                   image_shape=None, class_mapping=None):
    from ..vision.geometry import box_iou
    candidates = []
    used_tracks = False
    if tracks:
        for t in tracks:
            status = getattr(t, "status", "CONFIRMED")
            if status not in ("CONFIRMED", "TEMPORARILY_LOST"):
                continue
            conf = float(getattr(t, "confidence", 0.5))
            bbox = tuple(getattr(t, "bbox"))
            cx, cy = getattr(t, "center", ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
            rel = _relevance_for_class(getattr(t, "class_name", "obstacle"), conf, class_mapping)
            if RELEVANCE_RANK[rel] < RELEVANCE_RANK[min_relevance]:
                continue
            candidates.append(NavObstacle(
                obstacle_id=f"trk_{getattr(t, 'track_id', 0)}",
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                center=(float(cx), float(cy)),
                area_px=max(1, (bbox[2]-bbox[0])*(bbox[3]-bbox[1])),
                relevance=rel, source_class=getattr(t, "class_name", "unknown"),
                confidence=conf, track_id=int(getattr(t, "track_id", 0)),
            ))
            used_tracks = True
    if not used_tracks:
        for i, d in enumerate(detections or []):
            conf = float(getattr(d, "confidence", 0))
            name = getattr(d, "class_name", "obstacle")
            bbox = getattr(d, "bbox", None)
            if not bbox:
                continue
            rel = _relevance_for_class(name, conf, class_mapping)
            if RELEVANCE_RANK[rel] < RELEVANCE_RANK[min_relevance]:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            candidates.append(NavObstacle(
                obstacle_id=f"det_{i}", bbox=(x1, y1, x2, y2),
                center=((x1+x2)/2.0, (y1+y2)/2.0),
                area_px=max(1, (x2-x1)*(y2-y1)), relevance=rel,
                source_class=str(name), confidence=conf,
            ))
    candidates.sort(key=lambda o: RELEVANCE_RANK[o.relevance] * 10 + o.confidence, reverse=True)
    merged = []
    for o in candidates:
        absorbed = False
        for m in merged:
            if box_iou(o.bbox, m.bbox) >= merge_iou:
                x1 = min(o.bbox[0], m.bbox[0]); y1 = min(o.bbox[1], m.bbox[1])
                x2 = max(o.bbox[2], m.bbox[2]); y2 = max(o.bbox[3], m.bbox[3])
                m.bbox = (x1, y1, x2, y2)
                m.center = ((x1+x2)/2.0, (y1+y2)/2.0)
                m.area_px = max(1, (x2-x1)*(y2-y1))
                if RELEVANCE_RANK[o.relevance] > RELEVANCE_RANK[m.relevance]:
                    m.relevance = o.relevance
                m.confidence = max(m.confidence, o.confidence)
                absorbed = True
                break
        if not absorbed:
            merged.append(o)
    return merged

def obstacles_to_occupancy_mask(obstacles, height, width, min_relevance="MEDIUM"):
    mask = np.zeros((height, width), dtype=np.uint8)
    for o in obstacles:
        if RELEVANCE_RANK.get(o.relevance, 0) < RELEVANCE_RANK[min_relevance]:
            continue
        x1, y1, x2, y2 = o.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        mask[y1:y2, x1:x2] = 1
    return mask

def fusion_stats(raw_count, fused):
    return {
        "raw_detections": raw_count,
        "fused_obstacles": len(fused),
        "duplicate_reduction": max(0, raw_count - len(fused)),
        "by_relevance": {r: sum(1 for o in fused if o.relevance == r)
                         for r in ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")},
    }
