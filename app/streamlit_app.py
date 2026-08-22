"""Vision Robotics Analysis Lab — Streamlit Engineering UI"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.pipeline import AnalysisPipeline

st.set_page_config(
    page_title="Vision Robotics Analysis Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("VRAL")
    st.caption("Vision Robotics Analysis Lab")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
        help="JPG, PNG, WEBP or BMP",
    )

    st.subheader("Detection")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    min_area = st.slider("Min contour area", 20, 500, 80, 10)

    st.subheader("Visualization")
    show_free = st.checkbox("Free-space overlay", True)
    show_path = st.checkbox("Navigation path", True)

    st.subheader("Planning")
    run_planner = st.checkbox("Run image-space planner", True)
    cell_size = st.slider("Grid cell size (px)", 8, 32, 16, 4)

    st.markdown("---")
    analyze_btn = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("Detector: Classical CV · Navigation: image-space only")


def load_image(uploaded_file) -> np.ndarray:
    """Decode upload robustly. Prefer OpenCV; fall back to Pillow."""
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("Empty file (0 bytes). Try another image.")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

    if img is None or img.size == 0:
        raise ValueError("Decoded image is empty.")

    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_resource
def get_pipeline(min_area: int, conf: float, cell: int) -> AnalysisPipeline:
    return AnalysisPipeline(
        min_area=min_area,
        conf_threshold=conf,
        cell_size=cell,
        max_image_side=1280,
    )


st.title("Vision Robotics Analysis Lab")
st.markdown(
    "**Computer Vision · Scene · Risk · Image-space Navigation**"
)

tabs = st.tabs(
    ["Overview", "Vision", "Scene", "Navigation", "Brain", "Diagnostics", "Architecture"]
)

if "result" not in st.session_state:
    st.session_state.result = None
if "original_bgr" not in st.session_state:
    st.session_state.original_bgr = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# Preview upload immediately
if uploaded is not None:
    file_id = f"{uploaded.name}-{uploaded.size}"
    try:
        preview = load_image(uploaded)
        st.session_state.original_bgr = preview

        # Auto-run on new file, or when user clicks the button
        should_run = analyze_btn or (file_id != st.session_state.last_file_id)
        if should_run:
            st.session_state.last_file_id = file_id
            with st.spinner("Running analysis pipeline…"):
                pipeline = get_pipeline(min_area, conf_thresh, cell_size)
                st.session_state.result = pipeline.run(preview, run_planner=run_planner)
            st.success(f"Analyzed: {uploaded.name} ({preview.shape[1]}×{preview.shape[0]})")
    except Exception as e:
        st.session_state.result = None
        st.error(f"Image load / analysis failed: {e}")
        st.exception(e)

result = st.session_state.result
original_bgr = st.session_state.original_bgr

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detector", "Classical CV")
    c2.metric("Depth", "Unavailable")
    c3.metric("Tracking", "N/A")
    c4.metric("Mode", "Image Analysis")

    if original_bgr is None:
        st.info("Upload an image in the sidebar (JPG / PNG / WEBP / BMP).")
    elif result is None:
        st.warning("Image loaded but analysis has not completed. Click **Run analysis**.")
        st.image(bgr_to_rgb(original_bgr), caption="Preview", use_container_width=True)
    else:
        m = result.metrics()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Detections", m["detection_count"])
        k2.metric("Free space", f"{m['free_space_ratio']*100:.1f}%")
        k3.metric("Risk", m["risk_level"])
        k4.metric("Decision", m["decision"])
        k5.metric("Latency", f"{m['processing_time_ms']:.0f} ms")

with tabs[1]:
    st.subheader("Object Detection")
    if original_bgr is None:
        st.info("No image yet.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("**Original**")
            st.image(bgr_to_rgb(original_bgr), use_container_width=True)
        with b:
            st.markdown("**Annotated**")
            if result is not None:
                st.image(bgr_to_rgb(result.annotated_image), use_container_width=True)
            else:
                st.image(bgr_to_rgb(original_bgr), use_container_width=True)
        if result is not None and result.detections:
            st.dataframe(
                [d.to_dict() for d in result.detections],
                use_container_width=True,
            )
        elif result is not None:
            st.caption("No detections above threshold.")

with tabs[2]:
    st.subheader("Scene Understanding")
    if result is None:
        st.info("No analysis yet.")
    else:
        s = result.scene
        a, b, c, d = st.columns(4)
        a.metric("Objects", s.object_count)
        b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%")
        d.metric("Density", f"{s.obstacle_density*100:.1f}%")
        if show_free and s.free_space_mask is not None:
            st.image(bgr_to_rgb(result.free_space_overlay), use_container_width=True)
        for n in s.processing_notes:
            st.caption(n)
        st.json(s.to_dict())

with tabs[3]:
    st.subheader("Image-Space Navigation")
    st.markdown("Paths are **pixel trajectories**, not metric plans.")
    if result is None:
        st.info("No analysis yet.")
    else:
        if show_path:
            st.image(bgr_to_rgb(result.path_overlay), use_container_width=True)
            st.caption("Green = start · Red = goal · Cyan = path")
        if result.plan_comparison:
            st.table(
                [
                    {
                        "Algorithm": p.algorithm.upper(),
                        "Success": p.success,
                        "Length (px)": round(p.path_length_px, 1),
                        "Time (ms)": round(p.execution_time_ms, 2),
                        "Nodes": p.nodes_explored,
                    }
                    for p in result.plan_comparison
                ]
            )

with tabs[4]:
    st.subheader("Robot Brain")
    if result is None:
        st.info("No analysis yet.")
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
        for n in risk.notes:
            st.caption(n)

with tabs[5]:
    st.subheader("Diagnostics")
    if result is None:
        st.info("No analysis yet.")
    else:
        st.json(result.metrics())
        for n in result.notes:
            st.markdown(f"- {n}")

with tabs[6]:
    st.subheader("Architecture")
    st.markdown(
        """
```
Image → ClassicalDetector → SceneAnalyzer → RiskEngine
      → ImageSpacePlanner → DecisionEngine → Outputs
```
| Module | Status |
|--------|--------|
| Classical Detector | Implemented |
| Scene / Risk / Planner / Brain | Implemented |
| Depth / YOLO / ROS 2 | Future |
"""
    )
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")
