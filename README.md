# Vision Robotics Analysis Lab

**Computer Vision · Robotics · Numerical Systems · Image-space Navigation**

Engineering laboratory for monocular perception analysis and hypothetical image-space navigation.

> **YOLO is used as an external baseline and is not ARQTECH.** ARQTECH remains an experimental/future architecture. The repository does not claim training, accuracy, mAP, precision, recall, FPS, metric distance or real-world speed unless those quantities are actually measured under a documented methodology.

## Status matrix

| Capability | Status |
|---|---|
| Existing classical detector | IMPLEMENTED and always available |
| YOLO neural baseline | OPTIONAL; falls back safely when Ultralytics/weights are unavailable |
| Normalized detection interface | IMPLEMENTED |
| IoU tracker with lifecycle and stable IDs | IMPLEMENTED |
| Temporal smoothing | RAW, MOVING_AVERAGE and EXPONENTIAL |
| Navigation relevance / class mapping | EXPLICIT and configurable |
| Occupancy grid, cost map and image-space planner | IMPLEMENTED |
| Performance telemetry | MEASURED per run where available; otherwise `N/A` |
| Baseline comparison | IMPLEMENTED for same-frame detector output comparison |
| Experience Memory provenance | IMPLEMENTED; predictions remain predictions until human review |
| Camera calibration | PREPARATION ONLY; default `NOT CALIBRATED` |
| Metric depth / real-world velocity | NOT AVAILABLE |
| SLAM / EKF / ROS 2 | FUTURE |
| ARQTECH training | Experimental scaffold only; no claim of current superiority |

## Pipeline

```text
Image / Camera Frame
  → Preprocess
  → CURRENT DETECTOR or YOLO BASELINE
  → Adapter / NORMALIZED DETECTION
  → IoU TRACKER
  → RAW + TEMPORAL SMOOTHING
  → CLASS MAPPING / NAVIGATION RELEVANCE
  → GEOMETRY (IMAGE-SPACE)
  → OBSTACLE FUSION
  → OCCUPANCY / COST MAP
  → PATH PLANNER
  → ROBOT STATE
  → EXPERIENCE MEMORY
  → HUMAN REVIEW
  → VERSIONED DATASET
  → FUTURE ARQTECH TRAINING
```

The existing detector remains the fallback and the rest of the application never depends on Ultralytics result objects. Every normalized detection records class, confidence, bounding box, center, source model, model version, timestamp and frame ID. Tracks preserve raw and smoothed centers, lifecycle state, image-space velocity and position history.

## Install and run

```bash
git clone https://github.com/edu-moraess/vision-robotics-analysis-lab.git
cd vision-robotics-analysis-lab
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

YOLO is intentionally optional. To enable the neural baseline, install the optional dependency and provide valid weights through the Streamlit configuration:

```bash
pip install -r requirements-yolo.txt
```

If `ultralytics` is not installed, the selected mode displays `YOLO BASELINE: UNAVAILABLE` and continues with `CURRENT DETECTOR`. Invalid weights, unavailable GPU/CUDA and inference failures also degrade to the current detector instead of crashing the application.

## Streamlit controls

The sidebar exposes `CURRENT` and `YOLO_BASELINE`, model path, confidence threshold, IoU threshold, device, image size and maximum detections. Tracking can be enabled independently, and temporal smoothing supports `RAW`, `MOVING_AVERAGE` and `EXPONENTIAL` while preserving raw measurements. The `BASELINE COMPARISON` tab runs both detector adapters on the same preprocessed frame and reports detections per frame, average confidence and measured inference latency. It does not infer ground truth or declare a winner.

The perception HUD identifies the active model, model type and version. Bounding boxes display class, track ID and confidence. Navigation receives only classes explicitly present in the configured semantic mapping; unknown YOLO classes do not automatically become obstacles.

## Units and calibration

All distances and velocities are image-space by default. The application uses the labels `PIXEL MOTION` or `IMAGE-SPACE VELOCITY` and does not expose km/h. The calibration interface accepts future camera height, pitch, intrinsics, extrinsics, ground plane and homography inputs, but status remains `NOT CALIBRATED` until a valid calibration implementation is supplied. Metric coordinates and real-world speed must not be reported before that point.

## Experience Memory and active learning

Captured experiences store model name, model version, backend/type, frame provenance, detections, tracks, navigation state and temporal events. `model_prediction` is retained separately from `human_annotation`. The intended training path is:

```text
YOLO prediction → Experience Memory → Human review → Correction → Dataset → Training
```

YOLO predictions are never automatically converted into labels or training ground truth. ARQTECH training remains a future step and the repository does not claim that ARQTECH is currently trained or superior to YOLO.

## Recorded video and metrics

The recorded video lab can produce a report with frame count, confidence summaries, unique tracks, track persistence, person-track stability, duplicate detections, track switches, `STOPPED`, `PATH_BLOCKED` and `REPLANNING` transitions, processing time and model provenance. These are observations from the selected video and configuration. Baseline results must be preserved rather than overwritten when a before/after experiment is performed.

## Tests

```bash
pytest tests/ -v
```

The test suite covers the original detector, tracker, navigation and learning contracts plus YOLO adapter normalization, missing dependency fallback, invalid model handling, model provenance, track lifecycle, image-space history and temporal smoothing.

## Known limitations

This project remains a monocular, image-space engineering laboratory. It does not provide metric depth, physical robot control, camera calibration, SLAM, ROS 2 integration or a validated object-detection benchmark. Reported performance depends on the actual hardware, software versions, source video and sampling configuration.

## License

MIT
