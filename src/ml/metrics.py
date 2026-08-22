from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


METRIC_NAMES = ("mAP@50", "mAP@50-95", "precision", "recall", "f1")


@dataclass(frozen=True)
class DetectionMetricResult:
    status: str
    metrics: dict[str, float | None]
    counts: dict[str, int]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "metrics": dict(self.metrics),
            "counts": dict(self.counts),
            "notes": list(self.notes),
        }


def _xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in box]
    return x, y, x + width, y + height


def _iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(second)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class ConditionalDetectionMetricEvaluator:
    """Compute limited metrics only when real predictions and ground truth are supplied."""

    @staticmethod
    def not_measured(reason: str = "Real evaluated dataset and predictions were not supplied") -> DetectionMetricResult:
        return DetectionMetricResult(
            status="NOT MEASURED",
            metrics={name: None for name in METRIC_NAMES},
            counts={"tp": 0, "fp": 0, "fn": 0},
            notes=(reason, "No metric value is inferred from configuration or model existence."),
        )

    def evaluate(self, predictions: list[dict] | None = None,
                 ground_truth: list[dict] | None = None,
                 iou_threshold: float = 0.5) -> DetectionMetricResult:
        if not predictions or not ground_truth:
            return self.not_measured()
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        truth_by_image = {str(row.get("image_id")): row for row in ground_truth}
        tp = fp = fn = 0
        for prediction in predictions:
            image_id = str(prediction.get("image_id"))
            truth = truth_by_image.get(image_id, {"boxes": [], "labels": []})
            pred_boxes = prediction.get("boxes") or []
            pred_labels = prediction.get("labels") or []
            truth_boxes = truth.get("boxes") or []
            truth_labels = truth.get("labels") or []
            used_truth: set[int] = set()
            for p_box, p_label in zip(pred_boxes, pred_labels):
                candidates = [
                    (index, _iou(p_box, t_box))
                    for index, (t_box, t_label) in enumerate(zip(truth_boxes, truth_labels))
                    if index not in used_truth and str(t_label) == str(p_label)
                ]
                match = max(candidates, key=lambda pair: pair[1], default=(-1, 0.0))
                if match[0] >= 0 and match[1] >= iou_threshold:
                    tp += 1
                    used_truth.add(match[0])
                else:
                    fp += 1
            fn += max(0, len(truth_boxes) - len(used_truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return DetectionMetricResult(
            status="MEASURED",
            metrics={
                "mAP@50": None,
                "mAP@50-95": None,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            counts={"tp": tp, "fp": fp, "fn": fn},
            notes=(
                "Precision/recall/F1 measured from supplied predictions and ground truth.",
                "mAP is not measured by this evaluator.",
            ),
        )
