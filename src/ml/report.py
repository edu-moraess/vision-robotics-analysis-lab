from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class LearningReportGenerator:
    def __init__(self, reports_root="data/reports", datasets_root="data/datasets", registry_root="data/models"):
        self.reports_root = Path(reports_root)
        self.datasets_root = Path(datasets_root)
        self.registry_root = Path(registry_root)
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def generate(self, model_version="classical-cv-baseline", dataset_id=None,
                 experience_samples=None, experiment_config=None):
        models = []
        for idx in (self.registry_root / "registry.jsonl", self.registry_root / "models.jsonl"):
            if not idx.exists():
                continue
            for line in idx.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        models.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        model = next((m for m in models if model_version in (
            m.get("version"), m.get("model_version"), m.get("model_name"))), None)
        dataset = None
        if dataset_id:
            mp = self.datasets_root / dataset_id / "manifest.json"
            if mp.exists():
                dataset = json.loads(mp.read_text(encoding="utf-8"))
        exp = experience_samples or []
        status_counts = {}
        for sample in exp:
            status = sample.get("review_status", "pending")
            status_counts[status] = status_counts.get(status, 0) + 1
        metrics = (model or {}).get("metrics") or {}
        measured = {k: v for k, v in metrics.items() if v is not None}
        not_measured = [k for k in ("mAP@50", "mAP@50-95", "precision", "recall", "f1") if k not in measured]
        lifecycle_status = (model or {}).get("lifecycle_status") or (model or {}).get("status") or "NOT TRAINED"
        report = {
            "title": "MODEL LEARNING REPORT",
            "generated_at": time.time(),
            "model_summary": {
                "version": model_version,
                "record": model,
                "lifecycle_status": lifecycle_status,
                "trained": lifecycle_status in ("TRAINED", "VALIDATING", "VALIDATED", "ACTIVE"),
                "metrics_measured": measured,
                "metrics_not_measured": not_measured,
            },
            "experiment_config": experiment_config or {"status": "NOT PROVIDED"},
            "dataset_analysis": dataset or {"status": "NO_DATASET_SELECTED"},
            "experience_memory": {"total_listed": len(exp), "review_status_counts": status_counts},
            "learning_analysis": {
                "curves": "MEASURED ONLY IF PRESENT IN THE EXPERIMENT RECORD",
                "note": "Synthetic classification curves must not be reported as production detection metrics.",
            },
            "class_performance": "NOT MEASURED" if "precision" not in measured else measured.get("precision"),
            "error_analysis": "NOT MEASURED",
            "comparison": {
                "current_detector": "ACTIVE",
                "yolo": "EXTERNAL BASELINE / OPTIONAL",
                "groq": "EXTERNAL MULTIMODAL ANALYSIS / OPTIONAL",
                "arqtech": lifecycle_status,
            },
            "conclusion": "Insufficient evaluation data for performance claims." if not measured else "See metrics_measured; scope remains dataset-specific.",
            "honesty_notes": [
                "YOLO is an external baseline and is not ARQTECH.",
                "Groq is an external multimodal analysis layer and is not ARQTECH.",
                "Predictions are not ground truth until human review and annotation.",
            ],
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"arqtech_report_{model_version.replace('/', '_')}_{dataset_id or 'nodataset'}_{ts}"
        json_path = self.reports_root / f"{base}.json"
        html_path = self.reports_root / f"{base}.html"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        html_path.write_text(f"<html><body><pre>{json.dumps(report, indent=2, default=str)}</pre></body></html>", encoding="utf-8")
        report["export_json"] = str(json_path)
        report["export_html"] = str(html_path)
        return report
