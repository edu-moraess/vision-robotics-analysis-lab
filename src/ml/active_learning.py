"""Active learning prioritization — human review only, never auto-labels."""
from __future__ import annotations
from typing import Any, Dict, List

def rank_for_review(samples: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    ranked = []
    for s in samples:
        if s.get("review_status") not in (None, "pending"):
            continue
        unc = s.get("uncertainty_overall")
        unc = float(unc) if unc is not None else 0.5
        dets = s.get("detections") or []
        if dets:
            confs = [float(d.get("confidence", 0.5)) for d in dets if isinstance(d, dict)]
            mean_conf = sum(confs) / max(len(confs), 1)
        else:
            mean_conf = 0.0
            unc = max(unc, 0.6)
        score = 0.55 * unc + 0.45 * (1.0 - mean_conf)
        ranked.append({**s, "_review_priority": round(score, 4)})
    ranked.sort(key=lambda x: x["_review_priority"], reverse=True)
    return ranked[:limit]
