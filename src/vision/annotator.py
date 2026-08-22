"""Visualization helpers."""
from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from .detector import Detection

CLASS_COLORS = {
    "person": (80, 180, 255), "obstacle": (0, 165, 255),
    "wall": (180, 180, 180), "dynamic": (50, 50, 255), "default": (0, 220, 120),
}

def annotate_detections(image, detections, draw_labels=True, draw_centers=True):
    out = image.copy()
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, CLASS_COLORS["default"])
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if draw_centers:
            cv2.circle(out, (int(det.center[0]), int(det.center[1])), 3, color, -1)
        if draw_labels:
            label = f"{det.class_name.upper()}"
            if det.object_id is not None:
                label = f"{label} #{int(det.object_id):02d}"
            label = f"{label} {float(det.confidence):.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return out

def overlay_free_space(image, mask, alpha=0.35):
    out = image.copy()
    overlay = out.copy()
    if mask is not None and mask.ndim == 2:
        overlay[mask > 0] = (40, 180, 60)
        overlay[mask == 0] = (40, 40, 200)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

def draw_path(image, path_px, color=(0, 220, 255)):
    out = image.copy()
    if len(path_px) < 2:
        return out
    pts = np.array(path_px, dtype=np.int32)
    cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
    cv2.circle(out, path_px[0], 6, (0, 255, 0), -1)
    cv2.circle(out, path_px[-1], 6, (0, 0, 255), -1)
    return out
