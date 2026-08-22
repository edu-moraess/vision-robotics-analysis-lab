"""Image preprocessing with inspectable intermediate stages (classical CV only)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import cv2
import numpy as np
from ..core.timing import LatencyBreakdown, measure

@dataclass
class PreprocessResult:
    original: np.ndarray
    resized: np.ndarray
    gray: np.ndarray
    equalized: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    scale: float
    latency: LatencyBreakdown

    def stages_dict(self) -> Dict[str, np.ndarray]:
        return {"original": self.original, "resized": self.resized, "gray": self.gray,
                "equalized": self.equalized, "blurred": self.blurred, "edges": self.edges}

class Preprocessor:
    def __init__(self, max_side=1280, canny_low=50, canny_high=150, blur_ksize=5, use_clahe=True):
        self.max_side = max_side
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.use_clahe = use_clahe

    def run(self, image_bgr: np.ndarray) -> PreprocessResult:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image in Preprocessor")
        lat = LatencyBreakdown()
        original = image_bgr
        with measure("resize", lat):
            h, w = image_bgr.shape[:2]
            scale = 1.0
            longest = max(h, w)
            if longest > self.max_side:
                scale = self.max_side / float(longest)
                resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                resized = image_bgr.copy()
        with measure("gray", lat):
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        with measure("equalize", lat):
            if self.use_clahe:
                equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            else:
                equalized = cv2.equalizeHist(gray)
        with measure("blur", lat):
            blurred = cv2.GaussianBlur(equalized, (self.blur_ksize, self.blur_ksize), 0)
        with measure("edges", lat):
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return PreprocessResult(original, resized, gray, equalized, blurred, edges, scale, lat)
