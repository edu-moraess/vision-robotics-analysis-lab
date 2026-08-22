"""Vision Robotics Analysis Lab — Engineering Control Room UI."""
from __future__ import annotations
import io, sys, time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.pipeline import AnalysisPipeline
from src.camera import WebcamSource, IPCameraSource

st.set_page_config(page_title="Vision Robotics Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp { background-color: #0e1117; color: #e6edf3; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; font-family: ui-monospace, monospace; }
.status-on { color: #3fb950; font-weight: 600; }
.status-off { color: #8b949e; }
.status-warn { color: #d29922; }
</style>""", unsafe_allow_html=True)

def load_image(uploaded_file):
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("Empty file (0 bytes).")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        raise ValueError("Decoded image is empty.")
    return img

def bgr_to_rgb(img):
    if img is None: return img
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache_resource
def get_pipeline(min_area, conf, cell, tracking):
    return AnalysisPipeline(min_area=min_area, conf_threshold=conf, cell_size=cell, max_image_side=1280, enable_tracking=tracking)

with st.sidebar:
    st.markdown("## VISION ROBOTICS LAB")
    st.caption("Computer Vision · Robotics · Numerical Systems")
    st.markdown("---")
    mode = st.radio("INPUT MODE", ["Image Analysis", "Live Camera"], index=0)
    uploaded = None
    cam_index = 0
    ip_url = ""
    start_cam = stop_cam = False
    cam_src = "Webcam"
    if mode == "Image Analysis":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=False)
    else:
        cam_src = st.selectbox("Camera source", ["Webcam", "IP / RTSP URL"])
        if cam_src == "Webcam":
            cam_index = st.number_input("Device index", 0, 10, 0, 1)
        else:
            ip_url = st.text_input("Stream URL", placeholder="rtsp://... or http://...")
        c1, c2 = st.columns(2)
        start_cam = c1.button("START", use_container_width=True)
        stop_cam = c2.button("STOP", use_container_width=True)
        st.caption("Live camera needs local hardware. Cloud: usually unavailable.")
    st.markdown("---")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    min_area = st.slider("Min contour area", 20, 500, 80, 10)
    enable_tracking = st.checkbox("Tracking (live only)", value=(mode == "Live Camera"))
    run_planner = st.checkbox("Image-space planner", True)
    cell_size = st.slider("Grid cell (px)", 8, 32, 16, 4)
    show_free = st.checkbox("Free-space overlay", True)
    show_path = st.checkbox("Navigation path", True)
    show_stages = st.checkbox("Preprocess stages", False)
    st.markdown("---")
    analyze_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    st.caption("Backend: Classical CV · Depth/YOLO/ROS2: not implemented")

for key, default in [("result", None), ("original_bgr", None), ("last_file_id", None),
                     ("camera", None), ("live_running", False), ("fps", 0.0), ("frame_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

if mode == "Image Analysis" and uploaded is not None:
    file_id = f"{uploaded.name}-{uploaded.size}"
    try:
        preview = load_image(uploaded)
        st.session_state.original_bgr = preview
        if analyze_btn or (file_id != st.session_state.last_file_id):
            st.session_state.last_file_id = file_id
            with st.spinner("Pipeline running…"):
                pipe = get_pipeline(min_area, conf_thresh, cell_size, False)
                st.session_state.result = pipe.run(preview, run_planner=run_planner)
            st.success(f"{uploaded.name} · {preview.shape[1]}×{preview.shape[0]}")
    except Exception as e:
        st.session_state.result = None
        st.error(f"Load/analysis failed: {e}")

if mode == "Live Camera":
    if stop_cam and st.session_state.camera is not None:
        try: st.session_state.camera.stop()
        except Exception: pass
        st.session_state.camera = None
        st.session_state.live_running = False
    if start_cam:
        try:
            if st.session_state.camera is not None:
                st.session_state.camera.stop()
            if cam_src == "Webcam":
                cam = WebcamSource(device_index=int(cam_index))
            else:
                if not ip_url.strip():
                    raise ValueError("Provide a stream URL.")
                cam = IPCameraSource(url=ip_url.strip())
            cam.start()
            st.session_state.camera = cam
            st.session_state.live_running = True
            st.session_state.frame_count = 0
            st.session_state._live_pipe = get_pipeline(min_area, conf_thresh, cell_size, enable_tracking)
        except Exception as e:
            st.session_state.live_running = False
            st.session_state.camera = None
            st.error(f"Camera start failed: {e}")
    if st.session_state.live_running and st.session_state.camera is not None:
        cam = st.session_state.camera
        if not cam.is_available():
            st.warning("Camera offline.")
            st.session_state.live_running = False
        else:
            t0 = time.perf_counter()
            ok, frame = cam.read()
            if ok and frame is not None:
                pipe = st.session_state.get("_live_pipe") or get_pipeline(min_area, conf_thresh, cell_size, enable_tracking)
                try:
                    result = pipe.run(frame, run_planner=run_planner)
                    st.session_state.result = result
                    st.session_state.original_bgr = frame
                    dt = time.perf_counter() - t0
                    st.session_state.fps = (1.0 / dt) if dt > 0 else 0.0
                    st.session_state.frame_count += 1
                except Exception as e:
                    st.error(f"Frame pipeline error: {e}")
            else:
                st.warning("Dropped / invalid frame.")
            time.sleep(0.03)
            st.rerun()

result = st.session_state.result
original_bgr = st.session_state.original_bgr

st.markdown("# VISION ROBOTICS ANALYSIS LAB")
st.markdown("Computer Engineering · Computer Vision · Robotics · Image-space Navigation")
s1, s2, s3, s4, s5 = st.columns(5)
cam_on = st.session_state.live_running
s1.markdown("**CAMERA**  \n" + ("<span class='status-on'>● ONLINE</span>" if cam_on else "<span class='status-off'>○ OFFLINE</span>"), unsafe_allow_html=True)
s2.markdown("**PERCEPTION**  \n" + ("<span class='status-on'>● ACTIVE</span>" if result else "<span class='status-off'>○ IDLE</span>"), unsafe_allow_html=True)
s3.markdown("**TRACKING**  \n" + ("<span class='status-on'>● ACTIVE</span>" if result and result.tracking_active else "<span class='status-off'>○ N/A</span>"), unsafe_allow_html=True)
s4.markdown("**PLANNER**  \n" + ("<span class='status-on'>● READY</span>" if result and result.plan and result.plan.success else "<span class='status-warn'>○ NO PATH</span>" if result else "<span class='status-off'>○ IDLE</span>"), unsafe_allow_html=True)
s5.markdown("**DEPTH**  \n<span class='status-off'>○ NOT AVAILABLE</span>", unsafe_allow_html=True)

tabs = st.tabs(["MISSION CONTROL", "LIVE / FRAME", "PERCEPTION", "SCENE", "NAVIGATION", "ROBOT BRAIN", "DIAGNOSTICS", "SYSTEM"])

with tabs[0]:
    if original_bgr is None:
        st.info("Upload an image or start a camera in the sidebar.")
    else:
        left, right = st.columns(2)
        left.image(bgr_to_rgb(original_bgr), caption="INPUT FRAME", use_container_width=True)
        right.image(bgr_to_rgb(result.annotated_image if result else original_bgr), caption="ANNOTATED", use_container_width=True)
        if result is not None:
            m = result.metrics()
            ks = st.columns(6)
            ks[0].metric("OBJECTS", m["detection_count"])
            ks[1].metric("FREE SPACE", f"{m['free_space_ratio']*100:.1f}%")
            ks[2].metric("RISK", m["risk_level"])
            ks[3].metric("ACTION", m["decision"])
            ks[4].metric("LATENCY", f"{m['processing_time_ms']:.0f} ms")
            ks[5].metric("FPS" if cam_on else "TRACKS", f"{st.session_state.fps:.1f}" if cam_on else m.get("track_count", 0))

with tabs[1]:
    st.markdown("### LIVE VISION / FRAME VIEW")
    if mode == "Live Camera":
        st.caption(f"Running={st.session_state.live_running} · frames={st.session_state.frame_count} · FPS≈{st.session_state.fps:.1f} (wall-clock)")
    if result is None:
        st.info("No frame analyzed.")
    else:
        st.image(bgr_to_rgb(result.path_overlay if show_path else result.annotated_image), use_container_width=True)
        if result.tracks:
            st.dataframe([t.to_dict() for t in result.tracks], use_container_width=True)

with tabs[2]:
    st.markdown("### PERCEPTION PIPELINE")
    st.caption("Classical CV baseline — not a neural network.")
    if result is None or result.preprocess is None:
        st.info("No analysis yet.")
    else:
        if show_stages:
            stages = result.preprocess.stages_dict()
            cols = st.columns(5)
            for col, name in zip(cols, ["resized", "gray", "equalized", "blurred", "edges"]):
                img = stages.get(name)
                if img is not None:
                    col.image(bgr_to_rgb(img) if img.ndim == 3 else img, caption=name.upper(), use_container_width=True)
        a, b = st.columns(2)
        a.image(bgr_to_rgb(result.annotated_image), caption="DETECTIONS", use_container_width=True)
        if result.detections:
            b.dataframe([d.to_dict() for d in result.detections], use_container_width=True)

with tabs[3]:
    st.markdown("### SCENE MODEL")
    if result is None:
        st.info("No analysis yet.")
    else:
        s = result.scene
        a, b, c, d = st.columns(4)
        a.metric("Objects", s.object_count); b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%")
        d.metric("Density", f"{s.obstacle_density*100:.1f}%")
        if show_free and s.free_space_mask is not None:
            st.image(bgr_to_rgb(result.free_space_overlay), use_container_width=True)
        if result.occupancy is not None:
            st.caption(result.occupancy.label)
            st.write(f"Grid {result.occupancy.shape} · free_ratio={result.occupancy.free_ratio():.3f}")
        st.json(s.to_dict())

with tabs[4]:
    st.markdown("### IMAGE-SPACE NAVIGATION")
    st.markdown("Hypothetical pixel trajectories — **not** metric-world navigation.")
    if result is None:
        st.info("No analysis yet.")
    else:
        if show_path:
            st.image(bgr_to_rgb(result.path_overlay), use_container_width=True)
        if result.plan_comparison:
            st.table([{"Algorithm": p.algorithm.upper(), "Success": p.success,
                       "Length (px)": round(p.path_length_px, 1),
                       "Time (ms)": round(p.execution_time_ms, 2),
                       "Nodes": p.nodes_explored} for p in result.plan_comparison])

with tabs[5]:
    st.markdown("### ROBOT BRAIN")
    if result is None:
        st.info("No analysis yet.")
    else:
        dec, risk = result.decision, result.risk
        c1, c2, c3 = st.columns(3)
        c1.metric("ACTION", dec.action); c2.metric("CONFIDENCE", f"{dec.confidence:.2f}"); c3.metric("RISK", risk.level)
        st.markdown(f"**REASON**\n\n{dec.reason}")
        st.metric("Risk score", f"{risk.score:.3f}")
        for name, val in risk.contributors.items():
            st.progress(min(1.0, val / 0.40), text=f"{name}: +{val:.3f}")
        if result.uncertainty is not None:
            st.markdown("**UNCERTAINTY (heuristic, not calibrated probability)**")
            st.metric("Overall", f"{result.uncertainty.overall:.3f}")
            st.json(result.uncertainty.to_dict())

with tabs[6]:
    st.markdown("### DIAGNOSTICS")
    if result is None:
        st.info("No analysis yet.")
    else:
        st.json(result.metrics())
        if result.latency is not None:
            st.markdown("**Latency breakdown (ms)**")
            st.json(result.latency.to_dict())

with tabs[7]:
    st.markdown("### SYSTEM")
    import platform
    st.markdown(f"""
| Component | Status |
|-----------|--------|
| Python | {platform.python_version()} |
| OpenCV | {cv2.__version__} |
| Streamlit | {st.__version__} |
| Detector | Classical CV — IMPLEMENTED |
| Tracker (IoU) | IMPLEMENTED (live only) |
| Occupancy / Cost map | IMPLEMENTED (image-space) |
| A* / Dijkstra | IMPLEMENTED (image-space) |
| Risk / Decision / Uncertainty | IMPLEMENTED |
| Depth / YOLO / ROS 2 | NOT AVAILABLE / FUTURE |
""")
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")
