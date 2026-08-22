# Vision Robotics Analysis Lab

**Computer Vision · Scene Understanding · Risk · Image-space Navigation**

Engineering platform for monocular image analysis with a professional Streamlit interface.

**Repository:** https://github.com/edu-moraess/vision-robotics-analysis-lab

---

## Status

| Capability | Status |
|------------|--------|
| Classical object detection | **Implemented** |
| Free-space / scene metrics | **Implemented** |
| Transparent risk scoring | **Implemented** |
| Image-space A* / Dijkstra | **Implemented** |
| Decision engine (Brain) | **Implemented** |
| Streamlit UI | **Implemented** |
| Metric depth / YOLO / ROS 2 | Future |

Navigation outputs from a single RGB image are **image-space estimates**, not metric-world trajectories.

---

## Pipeline

```
Image → ClassicalDetector → SceneAnalyzer → RiskEngine
      → ImageSpacePlanner (A*/Dijkstra) → DecisionEngine → Outputs
```

---

## Install

```bash
git clone https://github.com/edu-moraess/vision-robotics-analysis-lab.git
cd vision-robotics-analysis-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app/streamlit_app.py
```

### Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io)
2. Login with GitHub **edu-moraess**
3. New app → `edu-moraess/vision-robotics-analysis-lab`
4. Branch: `main`
5. Main file: **`app/streamlit_app.py`**
6. Deploy

---

## Tests

```bash
pytest tests/ -v
```

---

## Layout

```
app/streamlit_app.py
src/vision/   # detector, scene, geometry, annotator
src/brain/    # risk_engine, decision_engine
src/planning/ # image_planner
src/core/     # AnalysisPipeline
tests/
```

## License

MIT
