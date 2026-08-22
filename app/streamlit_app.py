"""Vision Robotics Analysis Lab — Engineering Control Room UI."""
from __future__ import annotations
import io, sys, time, uuid
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.pipeline import AnalysisPipeline
from src.camera import WebcamSource, IPCameraSource, SmartphoneCameraSource, VideoFileSource
from src.learning import ExperienceMemory, FrameCache
from src.arqtech import ModelRegistry, ExperimentLog, describe_architecture
from src.arqtech.detector_interface import select_backend
from src.ml import DatasetBuilder, rank_for_review, LearningReportGenerator, TrainingConfig, save_training_config

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
    st.caption("Computer Vision · Robotics · ARQTECH Research")
    st.markdown("---")
    mode = st.radio("INPUT MODE", ["Image Analysis", "Live Camera"], index=0)
    uploaded = None
    cam_index = 0
    ip_url = ""
    start_cam = stop_cam = reconnect_cam = False
    cam_src = "Webcam"
    if mode == "Image Analysis":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    else:
        cam_src = st.selectbox("Camera source", ["Webcam", "Smartphone (network stream)", "IP / RTSP URL", "Video File"])
        if cam_src == "Webcam":
            cam_index = st.number_input("Device index", 0, 10, 0, 1)
        else:
            ph = "http://192.168.0.10:8080/video" if "Smartphone" in cam_src else ("/path/to/video.mp4" if "Video" in cam_src else "rtsp://...")
            ip_url = st.text_input("Stream URL / path", placeholder=ph)
        c1, c2 = st.columns(2)
        start_cam = c1.button("START", use_container_width=True)
        stop_cam = c2.button("STOP", use_container_width=True)
        reconnect_cam = st.button("RECONNECT", use_container_width=True)
    st.markdown("---")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    min_area = st.slider("Min contour area", 20, 500, 80, 10)
    enable_tracking = st.checkbox("Tracking (live only)", value=(mode == "Live Camera"))
    run_planner = st.checkbox("Image-space planner", True)
    cell_size = st.slider("Grid cell (px)", 8, 32, 16, 4)
    show_free = st.checkbox("Free-space overlay", True)
    show_path = st.checkbox("Navigation path", True)
    st.markdown("---")
    analyze_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    st.caption("ARQTECH scaffold — Classical CV is the active detector.")

for key, default in [("result", None), ("original_bgr", None), ("last_file_id", None),
                     ("camera", None), ("live_running", False), ("fps", 0.0), ("frame_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default
if "experience_memory" not in st.session_state:
    st.session_state.experience_memory = ExperienceMemory()
if "frame_cache" not in st.session_state:
    st.session_state.frame_cache = FrameCache(max_frames=30)

if mode == "Image Analysis" and uploaded is not None:
    file_id = f"{uploaded.name}-{uploaded.size}"
    try:
        preview = load_image(uploaded)
        st.session_state.original_bgr = preview
        if analyze_btn or (file_id != st.session_state.last_file_id):
            st.session_state.last_file_id = file_id
            with st.spinner("Pipeline running…"):
                st.session_state.result = get_pipeline(min_area, conf_thresh, cell_size, False).run(preview, run_planner=run_planner)
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
    if reconnect_cam and st.session_state.camera is not None:
        try:
            if hasattr(st.session_state.camera, "reconnect"):
                st.session_state.camera.reconnect()
            else:
                st.session_state.camera.stop(); st.session_state.camera.start()
            st.session_state.live_running = True
        except Exception as e:
            st.error(f"Reconnect failed: {e}")
    if start_cam:
        try:
            if st.session_state.camera is not None: st.session_state.camera.stop()
            if cam_src == "Webcam":
                cam = WebcamSource(device_index=int(cam_index))
            elif "Smartphone" in cam_src:
                if not ip_url.strip(): raise ValueError("Provide smartphone stream URL.")
                cam = SmartphoneCameraSource(url=ip_url.strip())
            elif "Video" in cam_src:
                if not ip_url.strip(): raise ValueError("Provide video path.")
                cam = VideoFileSource(path=ip_url.strip())
            else:
                if not ip_url.strip(): raise ValueError("Provide stream URL.")
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
            st.warning("Camera offline."); st.session_state.live_running = False
        else:
            t0 = time.perf_counter()
            packet = cam.read()
            if packet is not None and packet.image is not None:
                frame = packet.image
                pipe = st.session_state.get("_live_pipe") or get_pipeline(min_area, conf_thresh, cell_size, enable_tracking)
                try:
                    result = pipe.run(frame, run_planner=run_planner)
                    st.session_state.result = result
                    st.session_state.original_bgr = frame
                    st.session_state.frame_cache.push(frame, source=packet.source, frame_id=packet.frame_id)
                    dt = time.perf_counter() - t0
                    st.session_state.fps = (1.0 / dt) if dt > 0 else 0.0
                    st.session_state.frame_count += 1
                except Exception as e:
                    st.error(f"Frame pipeline error: {e}")
            else:
                st.warning("Dropped / invalid frame.")
            time.sleep(0.03); st.rerun()

result = st.session_state.result
original_bgr = st.session_state.original_bgr

st.markdown("# VISION ROBOTICS ANALYSIS LAB")
s1, s2, s3, s4, s5 = st.columns(5)
cam_on = st.session_state.live_running
s1.markdown("**CAMERA**  \n" + ("<span class='status-on'>● ONLINE</span>" if cam_on else "<span class='status-off'>○ OFFLINE</span>"), unsafe_allow_html=True)
s2.markdown("**PERCEPTION**  \n" + ("<span class='status-on'>● ACTIVE</span>" if result else "<span class='status-off'>○ IDLE</span>"), unsafe_allow_html=True)
s3.markdown("**MODEL**  \n<span class='status-on'>● CLASSICAL</span>", unsafe_allow_html=True)
s4.markdown("**ARQTECH**  \n<span class='status-off'>○ SCAFFOLD</span>", unsafe_allow_html=True)
s5.markdown("**TRAINING**  \n<span class='status-off'>○ IDLE</span>", unsafe_allow_html=True)

tabs = st.tabs(["MISSION CONTROL", "LIVE", "PERCEPTION", "SCENE", "NAVIGATION", "BRAIN", "REVIEW", "DATASET", "TRAINING", "ARQTECH", "DIAGNOSTICS", "SYSTEM"])

with tabs[0]:
    if original_bgr is None:
        st.info("Upload an image or start a camera.")
    else:
        left, right = st.columns(2)
        left.image(bgr_to_rgb(original_bgr), caption="INPUT", use_container_width=True)
        right.image(bgr_to_rgb(result.annotated_image if result else original_bgr), caption="ANNOTATED", use_container_width=True)
        if result is not None:
            m = result.metrics()
            ks = st.columns(5)
            ks[0].metric("OBJECTS", m["detection_count"])
            ks[1].metric("FREE SPACE", f"{m['free_space_ratio']*100:.1f}%")
            ks[2].metric("RISK", m["risk_level"])
            ks[3].metric("ACTION", m["decision"])
            ks[4].metric("LATENCY", f"{m['processing_time_ms']:.0f} ms")
            if st.button("STORE TO EXPERIENCE MEMORY"):
                sample = st.session_state.experience_memory.store(
                    image=result.annotated_image, camera_source="streamlit",
                    detections=[d.to_dict() for d in result.detections],
                    free_space_ratio=result.scene.estimated_free_space_ratio,
                    risk_score=result.risk.score, risk_level=result.risk.level,
                    decision=result.decision.action,
                    uncertainty_overall=result.uncertainty.overall if result.uncertainty else None,
                )
                st.success(f"Stored {sample.sample_id}" if sample else "Skipped.")

with tabs[1]:
    if result is None: st.info("No frame.")
    else: st.image(bgr_to_rgb(result.path_overlay if show_path else result.annotated_image), use_container_width=True)

with tabs[2]:
    if result is None: st.info("No analysis.")
    else:
        st.image(bgr_to_rgb(result.annotated_image), use_container_width=True)
        if result.detections: st.dataframe([d.to_dict() for d in result.detections], use_container_width=True)
        if getattr(result, "geometries", None):
            st.dataframe([g.to_dict() for g in result.geometries], use_container_width=True)

with tabs[3]:
    if result is None: st.info("No analysis.")
    else:
        s = result.scene
        a,b,c,d = st.columns(4)
        a.metric("Objects", s.object_count); b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%")
        d.metric("Density", f"{s.obstacle_density*100:.1f}%")
        if show_free and s.free_space_mask is not None:
            st.image(bgr_to_rgb(result.free_space_overlay), use_container_width=True)

with tabs[4]:
    if result is None: st.info("No analysis.")
    else:
        if show_path: st.image(bgr_to_rgb(result.path_overlay), use_container_width=True)
        if result.plan_comparison:
            st.table([{"Algorithm": p.algorithm.upper(), "Success": p.success, "Length (px)": round(p.path_length_px,1),
                       "Time (ms)": round(p.execution_time_ms,2), "Nodes": p.nodes_explored} for p in result.plan_comparison])

with tabs[5]:
    if result is None: st.info("No analysis.")
    else:
        dec, risk = result.decision, result.risk
        c1,c2,c3 = st.columns(3)
        c1.metric("ACTION", dec.action); c2.metric("CONF", f"{dec.confidence:.2f}"); c3.metric("RISK", risk.level)
        st.markdown(dec.reason)

with tabs[6]:
    st.markdown("### HUMAN REVIEW")
    st.caption("Predictions are NOT ground truth until accepted/corrected.")
    mem = st.session_state.experience_memory
    samples = mem.list_samples(limit=100)
    ranked = rank_for_review(samples, limit=20)
    st.write(f"Pending prioritized: {len(ranked)} / {len(samples)}")
    if ranked:
        choice = st.selectbox("Sample", [r.get("sample_id") for r in ranked])
        sel = next(r for r in ranked if r.get("sample_id") == choice)
        st.json({k: sel[k] for k in sel if k not in ("detections",)})
        if sel.get("detections"): st.dataframe(sel["detections"], use_container_width=True)
        c1,c2,c3 = st.columns(3)
        if c1.button("ACCEPT"): mem.set_review_status(choice, "accepted"); st.success("accepted")
        if c2.button("REJECT"): mem.set_review_status(choice, "rejected"); st.warning("rejected")
        if c3.button("CORRECT"): mem.set_review_status(choice, "corrected"); st.info("corrected")
    else:
        st.info("Store experiences from Mission Control first.")

with tabs[7]:
    st.markdown("### DATASET LAB")
    mem = st.session_state.experience_memory
    approved = [s for s in mem.list_samples(500) if s.get("review_status") in ("accepted", "corrected")]
    st.metric("Approved samples", len(approved))
    builder = DatasetBuilder()
    if st.button("Build new dataset version from approved"):
        try:
            man = builder.build_from_experiences(approved)
            st.success(f"Created {man.dataset_id}"); st.json(man.to_dict())
        except Exception as e:
            st.error(str(e))
    st.dataframe(builder.list_datasets(), use_container_width=True)

with tabs[8]:
    st.markdown("### TRAINING LAB")
    st.warning("Save config does NOT train. Metrics remain NOT MEASURED until a real run.")
    ds_ids = [d["dataset_id"] for d in DatasetBuilder().list_datasets()] or ["(none)"]
    ds_sel = st.selectbox("Dataset", ds_ids)
    mode_t = st.selectbox("Mode", ["BASELINE_INFERENCE", "FINE_TUNE", "FROM_SCRATCH"])
    model_name = st.selectbox("Model", ["CLASSICAL", "YOLO", "ARQTECH"])
    epochs = st.number_input("Epochs (planned)", 1, 500, 50)
    if st.button("Save training config (does not train)"):
        cfg = TrainingConfig(experiment_id=f"exp_{uuid.uuid4().hex[:8]}", model_name=model_name,
                             training_mode=mode_t, dataset_id=ds_sel if ds_sel != "(none)" else "none", epochs=int(epochs))
        path = save_training_config(cfg)
        st.success(f"Saved {path}")
        st.json({"status": "CONFIGURED_NOT_STARTED", "metrics": {}})

with tabs[9]:
    st.markdown("### ARQTECH LAB")
    st.warning("ARQTECH is SCAFFOLD — not trained. No mAP claimed.")
    st.json(describe_architecture())
    reg = ModelRegistry()
    st.dataframe(reg.list_models(), use_container_width=True)
    if st.button("GENERATE LEARNING REPORT"):
        gen = LearningReportGenerator()
        rep = gen.generate(experience_samples=st.session_state.experience_memory.list_samples(200))
        st.success(f"JSON: {rep.get('export_json')}")
        st.json(rep)
    st.caption("Inference remains Classical CV until ARQTECH has an ACTIVE checkpoint.")

with tabs[10]:
    if result is None: st.info("No analysis.")
    else:
        st.json(result.metrics())
        if result.latency is not None: st.json(result.latency.to_dict())

with tabs[11]:
    import platform
    st.markdown(f"""
| Component | Status |
|-----------|--------|
| Classical Detector | **ACTIVE** |
| ARQTECH | **SCAFFOLD** |
| Human Review / Dataset Builder | **IMPLEMENTED** |
| Training config | **IMPLEMENTED** (not executed) |
| YOLO / Depth / ROS 2 | FUTURE |
| Python | {platform.python_version()} |
""")
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")
