# Vision Robotics Analysis Lab

**Computer Vision · Robotics · Numerical Systems · Image-space Navigation**

Engineering laboratory for monocular perception analysis and hypothetical image-space navigation.

**Repository:** https://github.com/edu-moraess/vision-robotics-analysis-lab

---

## Status matrix (honest)

| Capability | Status |
|------------|--------|
| Classical object detection | **IMPLEMENTED** |
| Preprocessing stages | **IMPLEMENTED** |
| IoU tracker | **IMPLEMENTED** (live/temporal only) |
| Scene / free-space heuristics | **IMPLEMENTED** (image-space) |
| Occupancy grid + cost map | **IMPLEMENTED** (image-space) |
| A* / Dijkstra | **IMPLEMENTED** (pixel paths) |
| Risk engine | **IMPLEMENTED** |
| Uncertainty engine | **IMPLEMENTED** (heuristic, not calibrated) |
| Decision engine (Robot Brain) | **IMPLEMENTED** |
| Latency breakdown | **IMPLEMENTED** |
| Webcam / IP camera | **IMPLEMENTED** |
| Streamlit engineering UI | **IMPLEMENTED** |
| Metric depth | **NOT AVAILABLE** |
| YOLO | **NOT AVAILABLE** |
| SLAM / EKF / ROS 2 | **FUTURE** |

Navigation from a single RGB image is **image-space only**, not metric-world.

---

## Pipeline

```
Image / Camera Frame
  → Preprocess → ClassicalDetector → IoUTracker (live)
  → SceneAnalyzer → OccupancyGrid + CostMap
  → RiskEngine + UncertaintyEngine
  → ImageSpacePlanner (A* / Dijkstra)
  → DecisionEngine → Outputs
```

## Install & run

```bash
git clone https://github.com/edu-moraess/vision-robotics-analysis-lab.git
cd vision-robotics-analysis-lab
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Streamlit Cloud

- Main file: `app/streamlit_app.py`
- Login as repo owner
- **Note:** Live webcam is usually unavailable on Cloud; use Image Analysis mode.

## Tests

```bash
pytest tests/ -v
```

## Limitations

- Classical CV only
- Heuristic free-space / risk / uncertainty
- Pixel paths, not metric navigation
- No depth, no physical robot control

## License

MIT
