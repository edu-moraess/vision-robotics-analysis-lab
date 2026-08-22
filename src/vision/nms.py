"""Non-maximum suppression for Detection lists."""
from __future__ import annotations
from typing import List
from .detector import Detection
from .geometry import box_iou

def nms_detections(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
    kept: List[Detection] = []
    for det in ordered:
        drop = False
        for k in kept:
            iou = box_iou(det.bbox, k.bbox)
            same = det.class_name == k.class_name
            if iou >= iou_threshold and (same or iou >= 0.7):
                drop = True
                break
        if not drop:
            kept.append(det)
    return kept
