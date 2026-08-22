"""Navigation relevance separate from class labels. Pixel-space only."""
from __future__ import annotations
from typing import Dict, List

_DEFAULT: Dict[str, str] = {
    "person": "DYNAMIC_AGENT", "dynamic": "DYNAMIC_AGENT",
    "obstacle": "STATIC_OBSTACLE", "wall": "STRUCTURAL",
    "door": "POSSIBLE_OPENING", "window": "STRUCTURAL",
    "chair": "STATIC_OBSTACLE", "table": "STATIC_OBSTACLE",
    "car": "DYNAMIC_AGENT", "truck": "DYNAMIC_AGENT", "bicycle": "DYNAMIC_AGENT",
    "unknown": "UNKNOWN_RELEVANCE",
}

def navigation_relevance(class_name: str) -> str:
    if not class_name: return "UNKNOWN_RELEVANCE"
    return _DEFAULT.get(class_name.lower().strip(), "UNKNOWN_RELEVANCE")

def region_label(cx, cy, width, height) -> str:
    if width <= 0 or height <= 0: return "UNKNOWN"
    xr, yr = cx / width, cy / height
    horiz = "LEFT" if xr < 0.33 else ("RIGHT" if xr > 0.66 else "CENTER")
    vert = "FRONT" if yr > 0.45 else ("FAR" if yr < 0.25 else "MID")
    return f"{vert} / {horiz}"

def enrich_detections(detections, image_width, image_height):
    out = []
    for d in detections:
        if hasattr(d, "to_dict"):
            row = d.to_dict()
            name = getattr(d, "class_name", row.get("class_name", row.get("class", "unknown")))
            cx = getattr(d, "cx", row.get("cx", 0))
            cy = getattr(d, "cy", row.get("cy", 0))
        elif isinstance(d, dict):
            row = dict(d)
            name = row.get("class_name") or row.get("class") or "unknown"
            cx, cy = row.get("cx", 0), row.get("cy", 0)
        else:
            continue
        row["class_name"] = name
        row["navigation_relevance"] = navigation_relevance(name)
        row["position_label"] = region_label(float(cx), float(cy), image_width, image_height)
        row["distance"] = "NOT AVAILABLE"
        out.append(row)
    return out
