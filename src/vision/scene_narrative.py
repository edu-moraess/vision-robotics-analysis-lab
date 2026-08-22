"""Deterministic scene narrative — no LLM, no hallucinated objects."""
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional

def scene_inventory(detections):
    counts = Counter()
    for d in detections:
        if hasattr(d, "class_name"): counts[d.class_name] += 1
        elif isinstance(d, dict): counts[d.get("class_name") or d.get("class") or "unknown"] += 1
    return dict(counts)

def build_narrative(detections, free_space_ratio, path_available, decision_action=None, risk_level=None, enriched=None):
    lines = []
    n = len(detections)
    lines.append(f"{n} object(s) detected in the current frame.")
    inv = scene_inventory(detections)
    if inv:
        lines.append("Scene inventory: " + ", ".join(f"{k} × {v}" for k, v in sorted(inv.items(), key=lambda x: -x[1])) + ".")
    else:
        lines.append("No objects detected by the active model.")
    for row in (enriched or [])[:8]:
        name, pos, rel, conf = row.get("class_name", "object"), row.get("position_label", ""), row.get("navigation_relevance", ""), row.get("confidence", 0)
        if conf < 0.45:
            lines.append(f"Low-confidence detection: {name} ({conf:.0%}) at {pos}.")
        else:
            lines.append(f"{name.upper()} detected ({conf:.0%}) at {pos} — relevance: {rel}.")
    if free_space_ratio >= 0.55:
        lines.append(f"Estimated free space ahead is relatively open ({free_space_ratio:.0%} of lower ROI heuristic).")
    elif free_space_ratio >= 0.35:
        lines.append(f"Moderate free space in lower image region ({free_space_ratio:.0%} heuristic).")
    else:
        lines.append(f"Limited free space in lower image region ({free_space_ratio:.0%} heuristic).")
    lines.append("Image-space path planner found a traversable path." if path_available else "Image-space path planner did not find a clear path.")
    if decision_action:
        lines.append(f"Navigation decision: {decision_action}" + (f" (risk={risk_level})." if risk_level else "."))
    lines.append("Distance: NOT AVAILABLE (no calibrated depth).")
    return lines
