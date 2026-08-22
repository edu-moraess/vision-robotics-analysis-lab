from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class SmartCapturePolicy:
    uncertainty_threshold: float = 0.35
    confidence_threshold: float = 0.40
    cooldown_s: float = 5.0
    every_n_frames: int = 0
    capture_on_new_class: bool = True

class SmartCaptureState:
    def __init__(self):
        self.last_capture_ts = 0.0
        self.seen_classes = set()
        self.frame_counter = 0

def should_capture(policy, state, detections, uncertainty=None, force=False):
    state.frame_counter += 1
    now = time.time()
    if force:
        state.last_capture_ts = now; return True
    if policy.cooldown_s > 0 and (now - state.last_capture_ts) < policy.cooldown_s:
        return False
    if policy.every_n_frames > 0 and state.frame_counter % policy.every_n_frames == 0:
        state.last_capture_ts = now; return True
    if uncertainty is not None and uncertainty >= policy.uncertainty_threshold:
        state.last_capture_ts = now; return True
    confs, classes = [], []
    for d in detections or []:
        confs.append(float(getattr(d, "confidence", d.get("confidence", 1.0) if isinstance(d, dict) else 1.0)))
        classes.append(getattr(d, "class_name", d.get("class", "") if isinstance(d, dict) else ""))
    if confs and min(confs) < policy.confidence_threshold:
        state.last_capture_ts = now; return True
    if policy.capture_on_new_class:
        for c in classes:
            if c and c not in state.seen_classes:
                state.seen_classes.add(c); state.last_capture_ts = now; return True
            if c: state.seen_classes.add(c)
    return False
