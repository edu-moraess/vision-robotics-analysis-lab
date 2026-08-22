"""Experience memory — selective storage for human review. Not automatic ground truth."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

@dataclass
class ExperienceSample:
    sample_id: str
    timestamp: float
    camera_source: str
    image_path: str
    detections: List[dict]
    free_space_ratio: float
    risk_score: float
    risk_level: str
    decision: str
    uncertainty_overall: Optional[float]
    model_backend: str
    review_status: str = "pending"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

class ExperienceMemory:
    def __init__(self, root: str = "data/experience"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def _hash_image(self, image: np.ndarray) -> str:
        small = cv2.resize(image, (64, 64))
        return hashlib.sha1(small.tobytes()).hexdigest()[:16]

    def store(self, image, camera_source, detections, free_space_ratio, risk_score, risk_level,
              decision, uncertainty_overall=None, model_backend="classical_cv", min_uncertainty=0.0):
        if image is None or image.size == 0:
            return None
        if uncertainty_overall is not None and uncertainty_overall < min_uncertainty:
            return None
        h = self._hash_image(image)
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines()[-200:]:
                if h in line:
                    return None
        ts = time.time()
        sample_id = f"{int(ts)}_{h}"
        img_path = self.root / "images" / f"{sample_id}.jpg"
        cv2.imwrite(str(img_path), image)
        sample = ExperienceSample(
            sample_id=sample_id, timestamp=ts, camera_source=camera_source,
            image_path=str(img_path), detections=detections,
            free_space_ratio=free_space_ratio, risk_score=risk_score,
            risk_level=risk_level, decision=decision,
            uncertainty_overall=uncertainty_overall, model_backend=model_backend,
            notes=["Stored for human review — not automatic ground truth."],
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample.to_dict()) + "\n")
        return sample

    def list_samples(self, limit=50):
        if not self.index_path.exists():
            return []
        out = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(out))

    def set_review_status(self, sample_id, status):
        if status not in ("pending", "accepted", "corrected", "rejected") or not self.index_path.exists():
            return False
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        changed, new_lines = False, []
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                new_lines.append(line); continue
            if obj.get("sample_id") == sample_id:
                obj["review_status"] = status; changed = True
            new_lines.append(json.dumps(obj))
        if changed:
            self.index_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return changed

    def count(self):
        if not self.index_path.exists():
            return 0
        return sum(1 for line in self.index_path.open(encoding="utf-8") if line.strip())
