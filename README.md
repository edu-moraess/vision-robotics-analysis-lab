# Vision Robotics Analysis Lab

**Computer Vision · Robotics · Numerical Systems · Deep Learning · Image-space Navigation**

Engineering laboratory for modular monocular perception, robotics analysis and experimental model development.

> **YOLO is an external baseline and is not ARQTECH.**
>
> **Groq is an external multimodal analysis layer and is not ARQTECH.**

The repository does not claim training, accuracy, mAP, precision, recall, metric distance or real-world velocity unless those quantities are actually measured under a documented protocol.

## Architecture

```text
CAMERA / VIDEO / STREAM
        ↓
UNIVERSAL INPUT LAYER
        ↓
FRAME PACKET
(frame, frame_id, timestamp, source, resolution, fps, metadata)
        ↓
PERCEPTION ORCHESTRATOR
        ├── CURRENT DETECTOR
        ├── YOLO BASELINE
        ├── ARQTECH PyTorch MODEL
        └── GROQ MULTIMODAL ADVISORY
        ↓
PERCEPTION EVIDENCE / FUSION
        ↓
TRACKING → TEMPORAL SMOOTHING → GEOMETRY
        ↓
WORLD REPRESENTATION → OBSTACLE MODEL / FREE SPACE
        ↓
COST MAP → IMAGE-SPACE PATH PLANNER → ROBOT STATE
        ↓
EXPERIENCE MEMORY → HUMAN REVIEW → DATASET VERSION
        ↓
PYTORCH TRAINING → VALIDATION → MODEL REGISTRY
```

The perception layer consumes the canonical `FramePacket` contract and does not depend on whether the frame came from a webcam, smartphone, IP camera, RTSP stream, HTTP/MJPEG source or recorded video.

## Status matrix

| Area | Status |
|---|---|
| Universal input layer | IMPLEMENTED with canonical `FramePacket`, FPS and metadata |
| Webcam / smartphone / IP / RTSP / HTTP / video file | IMPLEMENTED through source adapters |
| FrameBuffer | IMPLEMENTED with push, latest-frame policy, drops and discarded-frame telemetry |
| Existing classical detector | IMPLEMENTED and always available |
| YOLO | OPTIONAL external neural baseline with safe fallback |
| Perception orchestrator | IMPLEMENTED with source evidence, latency and inference-time fusion |
| IoU tracking | IMPLEMENTED with lifecycle, stable IDs and image-space history |
| Temporal smoothing | RAW, MOVING_AVERAGE and EXPONENTIAL |
| Geometry and navigation | IMPLEMENTED in image-space |
| ARQTECH PyTorch architecture | IMPLEMENTED as a small modular experimental model |
| ARQTECH training | IMPLEMENTED for synthetic patch classification bootstrap only |
| ARQTECH object detection | NOT AVAILABLE without a reviewed detection checkpoint and postprocessing |
| ARQTECH validation | IMPLEMENTED for the supplied classification dataset/protocol only |
| Model lifecycle registry | IMPLEMENTED with controlled transitions |
| Groq multimodal layer | OPTIONAL, secret-backed, timeout-bounded and advisory |
| Experience Memory | IMPLEMENTED with model provenance, predictions, human review and external analysis |
| Dataset engineering | IMPLEMENTED with immutable versions and human-verified targets |
| Metric depth / camera calibration | NOT AVAILABLE; default `NOT CALIBRATED` |
| Real-world velocity / km/h | NOT AVAILABLE |
| SLAM / EKF / ROS 2 / physical control | FUTURE |

## Model identity

### ARQTECH

ARQTECH is the model owned by this repository and is implemented with PyTorch. Its current real training scope is a synthetic patch classification bootstrap. A checkpoint from that experiment is not an object detector and is not a production benchmark.

The lifecycle states are:

```text
NOT TRAINED → TRAINING → TRAINED → VALIDATING → VALIDATED → ACTIVE → ARCHIVED
```

A failed run may be retried from `FAILED`; invalid transitions are rejected by the registry. The Streamlit UI must not display `ACTIVE` or `VALIDATED` for a model that has not reached those states through an actual recorded process.

### YOLO

YOLO is an optional external neural baseline. The application preserves its actual model identity, version, weights and configuration. If Ultralytics or the selected weights are unavailable, the system reports `YOLO BASELINE: UNAVAILABLE` and continues with the existing detector.

### Groq

Groq is an optional external multimodal analysis layer. It can provide scene descriptions, ambiguity suggestions, semantic context and review questions. It is not a detector ground truth source, robot controller, geometry engine, metric measurement system or automatic ARQTECH training mechanism.

## Install and run

```bash
git clone https://github.com/edu-moraess/vision-robotics-analysis-lab.git
cd vision-robotics-analysis-lab
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

YOLO remains optional:

```bash
pip install -r requirements-yolo.txt
```

If a deployment intentionally omits PyTorch, the classical/YOLO/input portions can still be used, but ARQTECH modules will report unavailable rather than pretending to run. The standard requirements include PyTorch because the repository now contains a real, small training implementation.

## Groq configuration

Configure the key only through Streamlit Secrets:

```toml
GROQ_API_KEY = "your-runtime-secret"
GROQ_MODEL = "qwen/qwen3.6-27b"
```

The current official Groq vision documentation identifies `qwen/qwen3.6-27b` as a multimodal model and documents image input through a `chat.completions` message containing text plus a base64 `image_url` data URL.[1] The client also accepts runtime environment configuration for model, timeout, retries and token budget.

The key is never written to source code, README files, logs, screenshots, metrics, reports or Experience Memory. When the secret is missing, the UI displays `GROQ: DISABLED` and the rest of the pipeline continues.

## Input and FramePacket

All source adapters produce the same object:

| Field | Meaning |
|---|---|
| `frame` / `image` | BGR image array; `image` remains as a compatibility alias |
| `frame_id` | Monotonic source frame identifier when available |
| `timestamp` | Capture or read timestamp |
| `source` | Sanitized source label |
| `resolution` | `(width, height)` |
| `fps` | Source or measured FPS when available |
| `metadata` | Source-specific non-secret metadata |

`AnalysisPipeline.run_packet(packet)` is the preferred source-independent entry point. The buffer records capacity drops and older frames discarded by the latest-frame policy.

## Perception orchestration and fusion

The orchestrator records each source independently with model identity, version, weights, configuration, status, latency, raw detections and errors. Inference-time fusion may merge same-class boxes with sufficient IoU and retains the contributing source names and agreement count. This is not annotation and does not create ground truth.

Navigation receives only classes present in the explicit semantic mapping. Unknown YOLO classes and Groq suggestions do not automatically become obstacles.

## Tracking, geometry and units

Tracks use `CANDIDATE`, `CONFIRMED`, `TEMPORARILY_LOST` and `LOST` states and emit debounced temporal events. Raw centers remain available alongside smoothed centers. Motion is labeled `IMAGE-SPACE VELOCITY`; no km/h or metric speed is exposed.

The calibration interface is preparation-only. The default status is `NOT CALIBRATED`. Distances remain `IMAGE-SPACE` until a valid calibration implementation and documented parameters exist.

## Experience Memory and active learning

Experience Memory stores model name, model version, backend/type, frame provenance, detections, tracks, navigation state, events and optional external analysis. Predictions remain in `model_prediction`; human corrections remain in `human_annotation`.

The dataset builder uses human annotations for corrected samples and retains the original model prediction as provenance:

```text
MODEL PREDICTION
      ↓
EXPERIENCE MEMORY
      ↓
HUMAN REVIEW
      ↓
CORRECTION / ACCEPTANCE
      ↓
HUMAN-VERIFIED DATASET VERSION
      ↓
PYTORCH TRAINING
      ↓
VALIDATION
      ↓
MODEL REGISTRY
```

No prediction is automatically converted into a training target.

## ARQTECH modules

```text
src/arqtech/model/       ARQTECHModel
src/arqtech/backbone/    TinyConvBackbone
src/arqtech/heads/       ClassificationHead / DetectionHead
src/arqtech/loss/        Classification loss
src/arqtech/training/    Reusable fit engine
src/arqtech/validation/  Dataset-scoped validation
src/arqtech/inference/   Checkpoint loading and detector boundary
```

The detection head is an architectural extension, not a claim that reviewed detection labels or detection postprocessing already exist. `ArqtechDetector` refuses to present a classification checkpoint as an object detector.

## Deployment and dependencies

`requirements.txt` is the deployable dependency list and includes the runtime packages used by the repository, the official `groq` SDK and `ultralytics` for the external YOLO baseline. `requirements-yolo.txt` remains as a compatibility extra for older deployment scripts. `pyproject.toml` contains package metadata, test configuration and the same dependency intent without secrets. `.gitignore` excludes local environment files, credentials and secret/config directories.

The application can run without a Groq key and can fall back when Ultralytics, model weights or PyTorch are unavailable. No credential is committed. For Streamlit deployment, configure `GROQ_API_KEY` and optionally `GROQ_MODEL` through Streamlit Secrets only.

## Groq health and safety

The official Groq SDK is used for normal runtime calls. A compatibility HTTP path remains available for isolated tests and controlled fallback behavior. The public health status is one of `CONNECTED`, `NOT CONFIGURED`, `INVALID KEY`, `RATE LIMITED`, `ERROR` or `OFFLINE`; the UI exposes a non-destructive health check. Rate limits, timeout, latency and the last sanitized error are recorded without exposing the key.

Groq output is always labeled `AI GENERATED` and `NOT GROUND TRUTH`. It is advisory multimodal interpretation only: it cannot override detector output, create labels, control a robot, measure depth/speed or activate ARQTECH.

## Human review and dataset versioning

Experience Memory preserves the original `model_prediction` separately from `human_annotation`. The REVIEW panel supports `ACCEPT`, `EDIT`, `DELETE`, `ADD OBJECT`, `CHANGE CLASS` and `REJECT`, with reviewer, timestamp, action and provenance in `review_history`. The JSON editor accepts only explicit human annotations; it never converts a prediction into a label automatically.

Dataset versions are immutable. `DatasetBuilder` removes duplicate `image_hash` entries before splitting, groups samples by `session_id`/`source_identifier`/source, writes `groups.json`, stores split manifests and keeps the original prediction as provenance. Corrected samples use only `human_annotation` as their targets and labels are marked `HUMAN_VERIFIED`. A valid real detection loader additionally requires real images and numeric bounding boxes, so incomplete samples fail clearly instead of silently entering training.

## ARQTECH v0.3 detection readiness

ARQTECH v0.2 remains the implemented PyTorch synthetic patch-classification bootstrap. Its checkpoint is not an object detector. ARQTECH v0.3 is a separate `REAL OBJECT DETECTION` scaffold marked `EXPERIMENTAL / NOT TRAINED`; it includes raw detection-head contracts, an explicit activation gate and a reviewed-dataset loader, but no trained detection checkpoint or production post-processing claim.

The intended lifecycle is:

```text
ARCHITECTURE → BOOTSTRAP → REAL DATASET → HUMAN REVIEW →
DETECTION TRAINING → VALIDATION → BENCHMARK → PRODUCTION CANDIDATE
```

Training records contain epoch, train/validation loss, learning rate, duration, device and detection metric fields as `null` until a real execution writes them. The conditional evaluator returns `NOT MEASURED` without explicit evaluated predictions and ground truth. It measures only the supplied precision/recall/F1 at a declared IoU threshold; mAP remains `NOT MEASURED` in this scaffold.

## Tests

```bash
pytest tests/ -v
```

The suite covers input contracts, frame-buffer telemetry, detector normalization, YOLO fallback, perception fusion, tracking, smoothing, ARQTECH model forward/training/lifecycle, the v0.3 detection gate and reviewed dataset loader, conditional metrics, Groq disabled/success/error/rate-limit paths, Experience Memory review actions, grouped dataset versioning/deduplication, video processing and existing navigation behavior.

## Segmentation

The current segmentation implementation is `ContourSegmenter`, a deterministic OpenCV baseline that derives a local mask and contour from each detection bbox. It reports mask area and contour perimeter in pixels and is labeled `ESTIMATED`. It is not semantic neural segmentation and it is not ground truth. A future neural segmentation provider can implement the same interface without changing tracking or navigation contracts.

## Motion, trajectory and prediction

`MotionEngine` consumes confirmed temporal tracks and produces position, displacement, direction, acceleration, motion state and transition events. `TrajectoryEngine` stores timestamped per-track histories and produces a temporal heatmap marked `IMAGE-SPACE PROJECTION`. `ConstantVelocityPredictor` is a deterministic baseline; it is not an AI predictor and does not claim collision probability or physical motion.

The supported states are `STATIC`, `MOVING`, `APPROACHING`, `RECEDING`, `CROSSING` and `UNKNOWN`. `APPROACHING` and `RECEDING` are image-axis proxies until a valid calibration exists.

## Occupancy, risk and navigation

The pipeline now exposes numeric occupancy, semantic occupancy, risk zones and a navigation cost map. Semantic occupancy combines free-space heuristics with detections, masks and tracks and is explicitly projected image-space occupancy, not a 3D map. Risk is contextual: class prior, image position, confidence, motion and path availability are combined; a person is not automatically critical. A* and Dijkstra remain image-space planners and can consume the combined cost map.

`WorldModel` is the interface between perception and navigation. It carries objects, obstacles, free space, occupancy, trajectories, risk zones, current/alternative paths and simulation state while explicitly declaring that physical robot control is unavailable.

## Robot simulation

`RobotSimulation` is a deliberately bounded image/simulation-space kinematic visualization. It displays robot position, target, current path, alternative path, navigation state, obstacle count and risk-zone count. Its overlay is labeled `ROBOT SIMULATION`; it emits no actuator commands and must never be presented as a physical robot controller.

## Scientific limitations

This is a monocular engineering laboratory. It does not provide metric depth, validated camera calibration, physical robot control, SLAM, ROS 2 integration or a production object-detection benchmark. Synthetic classification accuracy is scoped to its synthetic hold-out and must not be reported as detection mAP or real-world performance.

## References

[1]: https://console.groq.com/docs/vision "Groq Docs — Images and Vision"

## License

MIT
