# Architectural Classification — Full Repository Audit

## Audit scope

The audit covered the repository tree, Python modules, Streamlit application, input sources, perception, YOLO adapter, ARQTECH/PyTorch, tracking, geometry, planning, robotics state, WorldModel, Experience Memory, ML dataset/report code, tests, configuration, requirements and documentation. The current baseline has **84 passing tests** on the published branch.

The first scan found that input, classical perception, YOLO fallback, tracking, image-space geometry, occupancy, risk, cost-map construction, planning, WorldModel, Experience Memory, dataset versioning, PyTorch bootstrap training and Groq integration already exist. It also found that dedicated segmentation, temporal state, motion, trajectory, heatmap, prediction, simulation and risk-zone modules do not yet exist as separate responsibilities.

## Classification matrix

| Component | Decision | Reason and next action |
|---|---|---|
| `app/streamlit_app.py` | REFACTOR | Keep the single entry point, but move new motion, heatmap and simulation calculations into services rather than adding business logic to UI. |
| `src/camera/*` | KEEP / IMPROVE | Source adapters and decoder diagnostics are useful. Improve metadata propagation, codec status and failure messages without coupling perception to source type. |
| `src/input/*` | KEEP | Resolver, buffer, security masking and smart capture are coherent with the lab mission. Extend only where new temporal consumers need packet provenance. |
| `FramePacket` | KEEP / IMPROVE | It is the correct cross-source boundary. Add optional packet-level metadata for capture timing and preserve backward-compatible aliases. |
| `ClassicalDetector` | KEEP | Existing local detector is the always-available fallback and should remain clearly labeled as the current detector. |
| `YoloDetector` | KEEP / EXPERIMENTAL | Retain as external baseline. Never attribute its outputs to ARQTECH. Add segmentation only through a separate optional source contract. |
| `PerceptionOrchestrator` | KEEP / IMPROVE | Correct boundary for independent evidence and fusion. Extend evidence to include segmentation without turning advisory output into ground truth. |
| `src/vision/detector.py` | IMPROVE | Detection contract needs optional mask/contour fields while keeping bbox-only sources valid. |
| `src/vision/geometry.py` | REFACTOR | Current bbox perimeter is not a real object perimeter. Preserve bbox geometry and add contour-derived metrics when a mask exists. |
| `src/vision/scene.py` | KEEP / IMPROVE | Heuristic free-space is useful image-space evidence. Rename/label it as projected heuristic occupancy and combine with semantic masks when available. |
| `src/vision/tracker.py` | KEEP / IMPROVE | Lifecycle and ID stability already exist. Feed motion/trajectory state into tracks and preserve temporal event debouncing. |
| `src/vision/temporal_smoothing.py` | KEEP | Raw and smoothed coordinates are valuable. Add an explicit temporal state service rather than hiding history in smoothing. |
| `src/vision/video_analysis.py` | REFACTOR | Retain report contract but add motion, trajectory, heatmap, segmentation and simulation summaries from measured results. |
| `src/vision/calibration.py` | EXPERIMENTAL / PLANNED | Keep the explicit `NOT CALIBRATED` state. Do not implement metric conversion without real calibration data. |
| `src/planning/occupancy.py` | KEEP / IMPROVE | Existing image-space occupancy and cost construction are useful. Add `UNKNOWN` preservation, semantic layers and risk/trajectory overlays. |
| `src/planning/obstacle_fusion.py` | KEEP / IMPROVE | It is the current perception-to-navigation bridge. Extend it with masks and risk context, not platform-specific logic. |
| `src/planning/image_planner.py` | KEEP / REFACTOR | A* and Dijkstra are real image-space planners. Centralize cost-map input and explicit replanning diagnostics. |
| `src/robotics/navigation_state.py` | KEEP / IMPROVE | State machine and transition events exist. Add hysteresis/debounce for repeated stops and distinguish `PATH_BLOCKED → REPLANNING → WAITING`. |
| `src/robotics/world_model.py` | KEEP / IMPROVE | Existing model is a useful interface but currently object/bbox-light. Extend with occupancy, paths, risk zones, trajectories and simulation state. |
| `src/brain/risk_engine.py` | KEEP / REFACTOR | Transparent deterministic scorer is valuable. Add per-object motion, trajectory intersection, occupancy and safety-margin inputs without making `PERSON` automatically critical. |
| `src/arqtech/*` | KEEP / EXPERIMENTAL | PyTorch model, training, validation, inference boundary and lifecycle registry are real but current task is synthetic patch classification. Keep detection unavailable until reviewed detection training exists. |
| `src/integrations/*` | KEEP | Groq is correctly isolated as external multimodal advisory analysis with secret handling and disabled fallback. |
| `src/learning/*` | KEEP / IMPROVE | Experience Memory and frame cache support active learning. Add masks, geometry, motion, trajectory, occupancy, risk and simulation provenance. |
| `src/ml/*` | KEEP / IMPROVE | Dataset, report and training configuration are mission-aligned. Add versioned segmentation/motion metadata and benchmark scopes. |
| Existing `src/brain`, `src/planning`, `src/robotics` locations | KEEP | They already provide coherent separation; moving them solely to match a nominal directory diagram would add risk without improving responsibility boundaries. |
| Dedicated segmentation package | PLANNED → IMPLEMENT | Missing in the baseline. Add a deterministic mask/contour provider first; keep neural segmentation optional and explicitly external/experimental. |
| Dedicated temporal state package | PLANNED → IMPLEMENT | Missing as a first-class service. Add per-track histories and transition-aware state. |
| Motion / trajectory / prediction packages | PLANNED → IMPLEMENT | Missing as separate services. Start with deterministic image-space calculations and constant-velocity prediction; future learned prediction remains planned. |
| Robot simulation package | PLANNED → IMPLEMENT | Missing. Add a clearly marked image/simulation-space kinematic visualization and never expose physical-control claims. |
| `docs/full_audit_raw.txt` and `docs/audit_gaps.txt` | MOVE / SUPPORTING | Keep as audit artifacts during this task, then place under a dedicated audit directory or retain only the synthesized report in the final branch. |
| Old scaffold wording | DEPRECATE | Replace UI and docs labels such as `SCAFFOLD` when they understate the modular implementation, while retaining honest `EXPERIMENTAL` and `NOT TRAINED` states. |
| Unused/dead modules | DO NOT REMOVE BLINDLY | No component is removed before reference analysis and regression tests. Candidates are documented first and removed only when no public/test/UI dependency remains. |
| Hardcoded secrets | REMOVE | The audit found no committed Groq key. Keep runtime-only `st.secrets` access and scan before publication. |
| Fabricated metrics or physical claims | REMOVE | Any unsupported mAP, speed, distance, depth, 3D, SLAM, LiDAR or radar claim remains excluded from code and reports. |

## Architectural decision

The existing repository is not discarded or flattened into a new directory tree. The correct strategy is a **responsibility-preserving extension**: perception remains independent of input source, image-space planning remains the safe navigation boundary, and new temporal/segmentation/simulation services are added as explicit modules with tests and provenance.

## Removal policy

No production component was removed in this audit because the existing tests and UI still depend on the current package layout. The only cleanup targets are obsolete wording, redundant report claims and any future orphan discovered after the second scan. The rule is to refactor useful code, merge duplicated responsibility, and remove only after reference verification.

## Validation categories

Every new output must be labeled as one of `DETECTED`, `ESTIMATED`, `INFERRED`, `PREDICTED`, `CALIBRATED`, `MEASURED`, `SIMULATED`, `EXTERNAL` or `EXPERIMENTAL`. The implementation must not collapse these categories into a single confidence or truth value.
