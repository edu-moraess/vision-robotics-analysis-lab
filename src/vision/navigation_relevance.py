from __future__ import annotations

from typing import Dict, List, Optional

from .perception_config import DEFAULT_CLASS_MAPPING


_SEMANTIC_DEFAULT: Dict[str, str] = {
    "person": "DYNAMIC_AGENT", "dynamic": "DYNAMIC_AGENT",
    "obstacle": "STATIC_OBSTACLE", "wall": "STRUCTURAL",
    "door": "POSSIBLE_OPENING", "window": "STRUCTURAL",
    "chair": "STATIC_OBSTACLE", "table": "STATIC_OBSTACLE",
    "car": "DYNAMIC_AGENT", "truck": "DYNAMIC_AGENT", "bicycle": "DYNAMIC_AGENT",
    "unknown": "UNKNOWN_RELEVANCE",
}


def navigation_relevance(class_name: str) -> str:
    """Backward-compatible semantic label, not a ground-truth obstacle label."""
    if not class_name:
        return "UNKNOWN_RELEVANCE"
    return _SEMANTIC_DEFAULT.get(class_name.lower().strip(), "UNKNOWN_RELEVANCE")


def navigation_priority(class_name: str, class_mapping: Optional[Dict[str, str]] = None) -> str:
    mapping = DEFAULT_CLASS_MAPPING if class_mapping is None else class_mapping
    return str(mapping.get((class_name or "").lower().strip(), "NONE")).upper()


def region_label(cx, cy, width, height) -> str:
    if width <= 0 or height <= 0:
        return "UNKNOWN"
    xr, yr = cx / width, cy / height
    horiz = "LEFT" if xr < 0.33 else ("RIGHT" if xr > 0.66 else "CENTER")
    vert = "FRONT" if yr > 0.45 else ("FAR" if yr < 0.25 else "MID")
    return f"{vert} / {horiz}"


def enrich_detections(detections, image_width, image_height, class_mapping=None):
    out = []
    for d in detections:
        if hasattr(d, "to_dict"):
            row = d.to_dict()
            name = getattr(d, "class_name", row.get("class_name", row.get("class", "unknown")))
            cx = getattr(d, "cx", row.get("cx", row.get("center", (0, 0))[0]))
            cy = getattr(d, "cy", row.get("cy", row.get("center", (0, 0))[1]))
        elif isinstance(d, dict):
            row = dict(d)
            name = row.get("class_name") or row.get("class") or "unknown"
            center = row.get("center", (0, 0))
            cx, cy = row.get("cx", center[0]), row.get("cy", center[1])
        else:
            continue
        row["class_name"] = name
        row["navigation_relevance"] = navigation_relevance(name)
        row["navigation_priority"] = navigation_priority(name, class_mapping)
        row["position_label"] = region_label(float(cx), float(cy), image_width, image_height)
        row["distance"] = "IMAGE-SPACE"
        row["calibration_status"] = "NOT CALIBRATED"
        out.append(row)
    return out
