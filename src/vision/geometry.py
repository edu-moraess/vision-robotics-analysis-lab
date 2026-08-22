"""Image-space geometric utilities."""
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))

def resize_keep_aspect(image, max_side=1280):
    h, w = image.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale

def lower_roi(image, ratio=0.45):
    h = image.shape[0]
    y0 = int(h * (1.0 - ratio))
    return image[y0:, :], y0
