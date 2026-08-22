"""Learning reports — empirical only; unmeasured metrics stay NOT MEASURED."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List, Optional

class LearningReportGenerator:
    def __init__(self, reports_root="data/reports", datasets_root="data/datasets", registry_root="data/arqtech/registry"):
        self.reports_root = Path(reports_root)
        self.datasets_root = Path(datasets_root)
        self.registry_root = Path(registry_root)
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def generate(self, model_version="classical-cv-baseline", dataset_id=None, experience_samples=None):
        models = []
        idx = self.registry_root / "models.jsonl"
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try: models.append(json.loads(line))
                    except json.JSONDecodeError: pass
        model = next((m for m in models if m.get("version") == model_version), None)
        dataset = None
        if dataset_id:
            mp = self.datasets_root / dataset_id / "manifest.json"
            if mp.exists():
                dataset = json.loads(mp.read_text(encoding="utf-8"))
        exp = experience_samples or []
        status_counts = {}
        for e in exp:
            st = e.get("review_status", "pending")
            status_counts[st] = status_counts.get(st, 0) + 1
        metrics = (model or {}).get("metrics") or {}
        measured = {k: v for k, v in metrics.items() if v is not None}
        not_measured = [k for k in ("mAP@50", "mAP@50-95", "precision", "recall") if k not in measured]
        report = {
            "title": "MODEL LEARNING REPORT",
            "generated_at": time.time(),
            "model_summary": {"version": model_version, "record": model, "trained": bool(measured),
                              "metrics_measured": measured, "metrics_not_measured": not_measured},
            "dataset_analysis": dataset or {"status": "NO_DATASET_SELECTED"},
            "experience_memory": {"total_listed": len(exp), "review_status_counts": status_counts},
            "learning_analysis": {"curves": "NOT MEASURED", "note": "No neural training curves until a real run."},
            "class_performance": "NOT MEASURED",
            "error_analysis": "NOT MEASURED",
            "comparison": {"classical_cv": "ACTIVE", "yolo": "NOT BUNDLED", "arqtech": "SCAFFOLD"},
            "conclusion": "Insufficient evaluation data for performance claims." if not measured else "See metrics_measured.",
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"arqtech_report_{model_version.replace('/', '_')}_{dataset_id or 'nodataset'}_{ts}"
        json_path = self.reports_root / f"{base}.json"
        html_path = self.reports_root / f"{base}.html"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        html_path.write_text(f"<html><body><pre>{json.dumps(report, indent=2)}</pre></body></html>", encoding="utf-8")
        report["export_json"] = str(json_path)
        report["export_html"] = str(html_path)
        return report
