# Vision Robotics Analysis Lab

**Computer Engineering · Computer Vision · Robotics · Numerical Systems**

Engineering laboratory for monocular perception, image-space navigation and decision support.

**Repository:** https://github.com/edu-moraess/vision-robotics-analysis-lab

---

## Status matrix

| Capability | Status |
|------------|--------|
| Classical detection | **IMPLEMENTED** |
| Preprocessing stages (CLAHE, edges) | **IMPLEMENTED** |
| Scene / free-space heuristics | **IMPLEMENTED** |
| Image-space occupancy + cost map | **IMPLEMENTED** |
| A* / Dijkstra (pixel grid) | **IMPLEMENTED** |
| Risk engine (transparent weights) | **IMPLEMENTED** |
| Decision engine (rule-based) | **IMPLEMENTED** |
| Webcam / IP camera abstraction | **IMPLEMENTED** |
| IoU tracker (temporal only) | **IMPLEMENTED** |
| Latency breakdown (perf_counter) | **IMPLEMENTED** |
| Streamlit engineering UI | **IMPLEMENTED** |
| Metric depth | **NOT AVAILABLE** |
| YOLO | **OPTIONAL / UNAVAILABLE** |
| EKF / SLAM / ROS 2 / physical control | **FUTURE** |

Navigation paths from RGB are **image-space estimates**. Risk scores are **heuristics**, not calibrated probabilities.

---

## Architecture

```
CAMERA / IMAGE
      ↓
PREPROCESS → DETECTION → TRACKING (live)
      ↓
SCENE + FREE-SPACE → OCCUPANCY + COST MAP
      ↓
RISK + DECISION → A*/DIJKSTRA (pixels)
```

---

## Install & run

```bash
git clone https://github.com/edu-moraess/vision-robotics-analysis-lab.git
cd vision-robotics-analysis-lab
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Streamlit Cloud: main file `app/streamlit_app.py`. Live webcam is typically unavailable on Cloud — use Image Analysis mode.

## Tests

```bash
pytest tests/ -v
```

## License

MIT
