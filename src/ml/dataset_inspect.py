from __future__ import annotations
from typing import Any, Dict, List

def inspect_manifest(manifest: Dict[str, Any]) -> List[str]:
    warnings = []
    n = manifest.get("sample_count", 0)
    if n < 10:
        warnings.append(f"Very small dataset ({n} samples) — results may not generalize.")
    dist = manifest.get("class_distribution") or {}
    if dist:
        total_ann = sum(dist.values()) or 1
        for cls, c in dist.items():
            if c < 3:
                warnings.append(f"Class '{cls}' has only {c} annotations.")
            if c / total_ann > 0.85:
                warnings.append(f"Possible class imbalance: '{cls}' is {100*c/total_ann:.0f}% of annotations.")
    if manifest.get("test_count", 0) == 0 and n > 2:
        warnings.append("Test split is empty.")
    if manifest.get("validation_count", 0) == 0 and n > 5:
        warnings.append("Validation split is empty.")
    return warnings
