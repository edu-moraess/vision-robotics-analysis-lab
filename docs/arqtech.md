# ARQTECH — Autonomous Robotics Perception Architecture

## Honest status

| Item | Status |
|------|--------|
| Research program name | DEFINED |
| Architecture plan | DOCUMENTED |
| Model registry | IMPLEMENTED |
| Experiment log | IMPLEMENTED |
| Detector interface | IMPLEMENTED |
| Trained weights | **NOT AVAILABLE** |
| mAP / precision / recall | **NOT AVAILABLE** |
| Live ARQTECH inference | **NOT AVAILABLE** |

**Active detector:** Classical CV baseline.

## BASELINE vs ARQTECH

| | BASELINE | ARQTECH |
|--|----------|---------|
| Type | OpenCV classical | Modular PyTorch research family |
| Status | **ACTIVE** | **EXPERIMENTAL** |
| Checkpoint | N/A | Classification bootstrap only; no detection checkpoint |
| Metrics | Latency only | Synthetic classification metrics only; no detection benchmark |

## Scientific principle

If ARQTECH underperforms baseline after a real training run, report it.
Never fabricate superiority.
