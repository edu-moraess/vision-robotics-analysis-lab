"""Experience Memory — selective storage. Predictions are NOT ground truth."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

@dataclass
class ExperienceSample:
    experience_id: str
    sample_id: str
    timestamp: float
    source_type: str
    source_identifier: str
    camera_source: str
    frame_id: Optional[int]
    image_path: str
    image_hash: str
    model_name: str
    model_version: str
    model_backend: str
    detections: List[dict]
    model_prediction: List[dict]
    human_annotation: Optional[List[dict]]
    free_space_ratio: float
    risk_score: float
    risk_level: str
    decision: str
    uncertainty_overall: Optional[float]
    capture_reason: str
    review_status: str = "pending"
    tracks: List[dict] = field(default_factory=list)
    navigation_state: Optional[dict] = None
    events: List[dict] = field(default_factory=list)
    external_analysis: Optional[dict] = None
    masks: List[dict] = field(default_factory=list)
    geometry: List[dict] = field(default_factory=list)
    motion: List[dict] = field(default_factory=list)
    trajectories: List[dict] = field(default_factory=list)
    risk: Optional[dict] = None
    occupancy: Optional[dict] = None
    simulation: Optional[dict] = None
    review_history: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

class ExperienceMemory:
    def __init__(self, root: str = "data/experience"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._seq_path = self.root / "sequence.txt"

    def _next_id(self):
        n = 1
        if self._seq_path.exists():
            try: n = int(self._seq_path.read_text().strip()) + 1
            except ValueError: n = 1
        self._seq_path.write_text(str(n))
        return f"EXP-MEM-{n:06d}"

    def _hash_image(self, image):
        return hashlib.sha1(cv2.resize(image, (64, 64)).tobytes()).hexdigest()[:16]

    def _hash_seen(self, h):
        if not self.index_path.exists(): return False
        return any(h in line for line in self.index_path.read_text(encoding="utf-8").splitlines()[-300:])

    def store(self, image, camera_source="unknown", detections=None, free_space_ratio=0.0,
              risk_score=0.0, risk_level="UNKNOWN", decision="UNKNOWN",
              uncertainty_overall=None, model_backend="classical_cv",
              model_name="classical-cv-baseline", model_version="baseline",
              source_type="UNKNOWN", source_identifier="", frame_id=None,
              capture_reason="MANUAL", min_uncertainty=0.0, skip_duplicate_hash=True,
              tracks=None, navigation_state=None, events=None, external_analysis=None,
              masks=None, geometry=None, motion=None, trajectories=None, risk=None,
              occupancy=None, simulation=None, notes=None):
        if image is None or image.size == 0: return None
        if uncertainty_overall is not None and min_uncertainty > 0 and uncertainty_overall < min_uncertainty:
            return None
        detections = detections or []
        h = self._hash_image(image)
        if skip_duplicate_hash and self._hash_seen(h): return None
        exp_id = self._next_id()
        img_path = self.root / "images" / f"{exp_id}.jpg"
        cv2.imwrite(str(img_path), image)
        sample = ExperienceSample(
            experience_id=exp_id, sample_id=exp_id, timestamp=time.time(),
            source_type=source_type or "UNKNOWN", source_identifier=source_identifier or camera_source,
            camera_source=camera_source, frame_id=frame_id, image_path=str(img_path), image_hash=h,
            model_name=model_name, model_version=model_version, model_backend=model_backend,
            detections=list(detections), model_prediction=list(detections), human_annotation=None,
            free_space_ratio=float(free_space_ratio), risk_score=float(risk_score),
            risk_level=str(risk_level), decision=str(decision),
            uncertainty_overall=uncertainty_overall, capture_reason=capture_reason, review_status="pending",
            tracks=list(tracks or []), navigation_state=navigation_state, events=list(events or []),
            external_analysis=external_analysis,
            masks=list(masks or []), geometry=list(geometry or []), motion=list(motion or []),
            trajectories=list(trajectories or []), risk=risk, occupancy=occupancy,
            simulation=simulation, notes=list(notes or []),
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample.to_dict()) + "\n")
        return sample

    def list_samples(self, limit=100, review_status=None):
        if not self.index_path.exists(): return []
        rows = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if review_status and row.get("review_status") != review_status: continue
            rows.append(row)
        return list(reversed(rows[-limit:]))

    def get(self, experience_id):
        for s in self.list_samples(10000):
            if s.get("experience_id") == experience_id or s.get("sample_id") == experience_id:
                return s
        return None

    def set_review_status(self, experience_id, status, human_annotation=None, notes=None):
        if status not in ("pending", "accepted", "corrected", "rejected"):
            raise ValueError(status)
        if not self.index_path.exists(): return False
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        out, found = [], False
        for line in lines:
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError:
                out.append(line); continue
            if row.get("experience_id") == experience_id or row.get("sample_id") == experience_id:
                row["review_status"] = status
                if human_annotation is not None: row["human_annotation"] = human_annotation
                if notes: row.setdefault("notes", []).append(notes)
                row.setdefault("review_history", []).append({
                    "timestamp": time.time(), "action": status.upper(),
                    "review_status": status, "human_annotation_present": human_annotation is not None,
                })
                found = True
            out.append(json.dumps(row))
        if found:
            self.index_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return found

    def apply_review(self, experience_id, action, annotations=None, reviewer="human", notes=None):
        action = str(action or "").upper().replace(" ", "_")
        action_to_status = {
            "ACCEPT": "accepted", "EDIT": "corrected", "DELETE": "corrected",
            "ADD_OBJECT": "corrected", "CHANGE_CLASS": "corrected", "REJECT": "rejected",
        }
        if action not in action_to_status:
            raise ValueError(f"Unsupported review action: {action}")
        if not self.index_path.exists():
            return False
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        out, found = [], False
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                out.append(line)
                continue
            if row.get("experience_id") == experience_id or row.get("sample_id") == experience_id:
                status = action_to_status[action]
                if action == "DELETE":
                    annotations = [] if annotations is None else [
                        a for a in (row.get("human_annotation") or row.get("detections") or [])
                        if a not in annotations
                    ]
                if annotations is not None:
                    row["human_annotation"] = list(annotations)
                row["review_status"] = status
                row.setdefault("review_history", []).append({
                    "timestamp": time.time(), "action": action, "reviewer": reviewer,
                    "annotation_count": len(row.get("human_annotation") or []),
                    "notes": notes or "",
                })
                if notes:
                    row.setdefault("notes", []).append(notes)
                found = True
            out.append(json.dumps(row))
        if found:
            self.index_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return found

    def count(self):
        return len(self.list_samples(100000))

    def summary(self):
        samples = self.list_samples(100000)
        counts = {"pending": 0, "accepted": 0, "corrected": 0, "rejected": 0}
        for s in samples:
            st = s.get("review_status", "pending")
            counts[st] = counts.get(st, 0) + 1
        return {"total": len(samples), **counts,
                "training_ready": counts["accepted"] + counts["corrected"]}
