"""Vision Robotics Analysis Lab — Streamlit Engineering UI"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.pipeline import AnalysisPipeline

st.set_page_config(page_title="Vision Robotics Analysis Lab", page_icon="◈", layout="wide")

with st.sidebar:
    st.title("VRAL")
    st.caption("Vision Robotics Analysis Lab")
    st.markdown("---")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    min_area = st.slider("Min contour area", 20, 500, 80, 10)
    show_free = st.checkbox("Free-space overlay", True)
    show_path = st.checkbox("Navigation path", True)
    run_planner = st.checkbox("Run image-space planner", True)
    cell_size = st.slider("Grid cell size (px)", 8, 32, 16, 4)
    st.caption("Detector: Classical CV · Navigation: image-space only")

def load_image(file):
    arr = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image.")
    return img

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache_resource
def get_pipeline(min_area, conf, cell):
    return AnalysisPipeline(min_area=min_area, conf_threshold=conf, cell_size=cell)

st.title("Vision Robotics Analysis Lab")
st.markdown("**Computer Vision · Scene · Risk · Image-space Navigation**")

tabs = st.tabs(["Overview", "Vision", "Scene", "Navigation", "Brain", "Diagnostics", "Architecture"])

if "result" not in st.session_state:
    st.session_state.result = None
if "original_bgr" not in st.session_state:
    st.session_state.original_bgr = None

if uploaded is not None:
    try:
        original = load_image(uploaded)
        st.session_state.original_bgr = original
        with st.spinner("Analyzing…"):
            st.session_state.result = get_pipeline(min_area, conf_thresh, cell_size).run(original, run_planner)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.session_state.result = None

result = st.session_state.result
original_bgr = st.session_state.original_bgr

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detector", "Classical CV")
    c2.metric("Depth", "Unavailable")
    c3.metric("Tracking", "N/A")
    c4.metric("Mode", "Image Analysis")
    if result is None:
        st.info("Upload an image in the sidebar.")
    else:
        m = result.metrics()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Detections", m["detection_count"])
        k2.metric("Free space", f"{m['free_space_ratio']*100:.1f}%")
        k3.metric("Risk", m["risk_level"])
        k4.metric("Decision", m["decision"])
        k5.metric("Latency", f"{m['processing_time_ms']:.0f} ms")

with tabs[1]:
    if result is None or original_bgr is None:
        st.info("No image yet.")
    else:
        a, b = st.columns(2)
        a.image(bgr_to_rgb(original_bgr), caption="Original", use_container_width=True)
        b.image(bgr_to_rgb(result.annotated_image), caption="Annotated", use_container_width=True)
        if result.detections:
            st.dataframe([d.to_dict() for d in result.detections], use_container_width=True)

with tabs[2]:
    if result is None:
        st.info("No image yet.")
    else:
        s = result.scene
        a, b, c, d = st.columns(4)
        a.metric("Objects", s.object_count)
        b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%")
        d.metric("Density", f"{s.obstacle_density*100:.1f}%")
        if show_free and s.free_space_mask is not None:
            st.image(bgr_to_rgb(result.free_space_overlay), use_container_width=True)
        st.json(s.to_dict())

with tabs[3]:
    st.markdown("Paths are **pixel trajectories**, not metric plans.")
    if result is None:
        st.info("No image yet.")
    else:
        if show_path:
            st.image(bgr_to_rgb(result.path_overlay), use_container_width=True)
        if result.plan_comparison:
            st.table([{"Algorithm": p.algorithm.upper(), "Success": p.success,
                       "Length (px)": round(p.path_length_px, 1),
                       "Time (ms)": round(p.execution_time_ms, 2),
                       "Nodes": p.nodes_explored} for p in result.plan_comparison])

with tabs[4]:
    if result is None:
        st.info("No image yet.")
    else:
        dec, risk = result.decision, result.risk
        c1, c2, c3 = st.columns(3)
        c1.metric("Action", dec.action)
        c2.metric("Confidence", f"{dec.confidence:.2f}")
        c3.metric("Risk", risk.level)
        st.markdown(f"**Reason:** {dec.reason}")
        st.metric("Risk score", f"{risk.score:.3f}")
        for name, val in risk.contributors.items():
            st.progress(min(1.0, val / 0.40), text=f"{name}: +{val:.3f}")

with tabs[5]:
    if result is None:
        st.info("No image yet.")
    else:
        st.json(result.metrics())

with tabs[6]:
    st.markdown("""
```
Image → ClassicalDetector → SceneAnalyzer → RiskEngine
      → ImageSpacePlanner → DecisionEngine → Outputs
```
| Module | Status |
|--------|--------|
| Classical Detector | Implemented |
| Scene / Risk / Planner / Brain | Implemented |
| Depth / YOLO / ROS 2 | Future |
""")
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")
