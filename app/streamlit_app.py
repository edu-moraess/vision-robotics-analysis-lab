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
from src.arqtech import ModelRegistry, describe_architecture
from src.ml import DatasetBuilder, rank_for_review, LearningReportGenerator, TrainingConfig, save_training_config, inspect_manifest
from src.input import InputManager, InputDescriptor, SourceType, SmartCapturePolicy, mask_url

st.set_page_config(page_title="Vision Robotics Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp { background-color: #0e1117; color: #e6edf3; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; font-family: ui-monospace, monospace; }
.status-on { color: #3fb950; font-weight: 600; }
.status-off { color: #8b949e; }
</style>""", unsafe_allow_html=True)

def load_image(uploaded_file):
    data = uploaded_file.getvalue()
    if not data: raise ValueError("Empty file.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0: raise ValueError("Decoded image empty.")
    return img

def bgr_to_rgb(img):
    if img is None: return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache_resource
def get_pipeline(min_area, conf, cell, tracking):
    return AnalysisPipeline(min_area=min_area, conf_threshold=conf, cell_size=cell, max_image_side=1280, enable_tracking=tracking)

with st.sidebar:
    st.markdown("## VISION ROBOTICS LAB")
    mode = st.radio("INPUT MODE", ["Image Analysis", "Live Camera"], index=0)
    uploaded = None
    cam_index, ip_url = 0, ""
    start_cam = stop_cam = reconnect_cam = False
    cam_src = "Webcam"
    if mode == "Image Analysis":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    else:
        cam_src = st.selectbox("Camera source", ["Webcam", "Smartphone (network stream)", "IP / RTSP URL", "Video File"])
        if cam_src == "Webcam":
            cam_index = st.number_input("Device index", 0, 10, 0, 1)
        else:
            ip_url = st.text_input("Stream URL / path", placeholder="http://192.168.0.10:8080/video")
        c1, c2 = st.columns(2)
        start_cam = c1.button("START", use_container_width=True)
        stop_cam = c2.button("STOP", use_container_width=True)
        reconnect_cam = st.button("RECONNECT", use_container_width=True)
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    min_area = st.slider("Min contour area", 20, 500, 80, 10)
    enable_tracking = st.checkbox("Tracking (live only)", value=(mode == "Live Camera"))
    run_planner = st.checkbox("Image-space planner", True)
    cell_size = st.slider("Grid cell (px)", 8, 32, 16, 4)
    show_path = st.checkbox("Navigation path", True)
    analyze_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    st.caption("Active: Classical CV · ARQTECH: scaffold · Inference ≠ Training")

for key, default in [("result", None), ("original_bgr", None), ("last_file_id", None),
                     ("camera", None), ("live_running", False), ("fps", 0.0), ("frame_count", 0)]:
    if key not in st.session_state: st.session_state[key] = default
if "experience_memory" not in st.session_state:
    st.session_state.experience_memory = ExperienceMemory()
if "frame_cache" not in st.session_state:
    st.session_state.frame_cache = FrameCache(max_frames=30)
if "input_manager" not in st.session_state:
    st.session_state.input_manager = InputManager()

if mode == "Image Analysis" and uploaded is not None:
    file_id = f"{uploaded.name}-{uploaded.size}"
    try:
        preview = load_image(uploaded)
        st.session_state.original_bgr = preview
        if analyze_btn or file_id != st.session_state.last_file_id:
            st.session_state.last_file_id = file_id
            with st.spinner("Pipeline…"):
                st.session_state.result = get_pipeline(min_area, conf_thresh, cell_size, False).run(preview, run_planner=run_planner)
    except Exception as e:
        st.session_state.result = None
        st.error(str(e))

if mode == "Live Camera":
    if stop_cam and st.session_state.camera is not None:
        try: st.session_state.camera.stop()
        except Exception: pass
        st.session_state.camera = None; st.session_state.live_running = False
    if reconnect_cam and st.session_state.camera is not None:
        try:
            if hasattr(st.session_state.camera, "reconnect"): st.session_state.camera.reconnect()
            else: st.session_state.camera.stop(); st.session_state.camera.start()
            st.session_state.live_running = True
        except Exception as e: st.error(str(e))
    if start_cam:
        try:
            if st.session_state.camera is not None: st.session_state.camera.stop()
            if cam_src == "Webcam": cam = WebcamSource(device_index=int(cam_index))
            elif "Smartphone" in cam_src: cam = SmartphoneCameraSource(url=ip_url.strip())
            elif "Video" in cam_src: cam = VideoFileSource(path=ip_url.strip())
            else: cam = IPCameraSource(url=ip_url.strip())
            cam.start(); st.session_state.camera = cam; st.session_state.live_running = True
            st.session_state._live_pipe = get_pipeline(min_area, conf_thresh, cell_size, enable_tracking)
        except Exception as e:
            st.session_state.live_running = False; st.session_state.camera = None; st.error(str(e))
    if st.session_state.live_running and st.session_state.camera is not None:
        if not st.session_state.camera.is_available():
            st.warning("Camera offline"); st.session_state.live_running = False
        else:
            t0 = time.perf_counter()
            packet = st.session_state.camera.read()
            if packet is not None and packet.image is not None:
                try:
                    result = st.session_state._live_pipe.run(packet.image, run_planner=run_planner)
                    st.session_state.result = result; st.session_state.original_bgr = packet.image
                    st.session_state.fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
                    st.session_state.frame_count += 1
                except Exception as e: st.error(str(e))
            time.sleep(0.03); st.rerun()

result, original_bgr = st.session_state.result, st.session_state.original_bgr
st.markdown("# VISION ROBOTICS ANALYSIS LAB")
s1,s2,s3,s4,s5 = st.columns(5)
cam_on = st.session_state.live_running
s1.markdown("**CAMERA**  \n" + ("<span class='status-on'>● ONLINE</span>" if cam_on else "<span class='status-off'>○ OFFLINE</span>"), unsafe_allow_html=True)
s2.markdown("**PERCEPTION**  \n" + ("<span class='status-on'>● ACTIVE</span>" if result else "<span class='status-off'>○ IDLE</span>"), unsafe_allow_html=True)
s3.markdown("**MODEL**  \n<span class='status-on'>● CLASSICAL</span>", unsafe_allow_html=True)
s4.markdown("**ARQTECH**  \n<span class='status-off'>○ SCAFFOLD</span>", unsafe_allow_html=True)
s5.markdown("**LEARN**  \n<span class='status-off'>○ LOOP</span>", unsafe_allow_html=True)

tabs = st.tabs(["MISSION CONTROL", "LIVE", "VIDEO INPUT", "PERCEPTION", "SCENE", "NAVIGATION", "BRAIN", "REVIEW", "DATASET", "TRAINING", "ARQTECH", "DIAGNOSTICS", "SYSTEM"])

with tabs[0]:
    if original_bgr is None: st.info("Upload an image or start a camera.")
    else:
        a,b = st.columns(2)
        a.image(bgr_to_rgb(original_bgr), caption="INPUT", use_container_width=True)
        b.image(bgr_to_rgb(result.annotated_image if result else original_bgr), caption="ANNOTATED", use_container_width=True)
        if result is not None:
            m = result.metrics()
            ks = st.columns(5)
            ks[0].metric("OBJECTS", m["detection_count"]); ks[1].metric("FREE SPACE", f"{m['free_space_ratio']*100:.1f}%")
            ks[2].metric("RISK", m["risk_level"]); ks[3].metric("ACTION", m["decision"])
            ks[4].metric("LATENCY", f"{m['processing_time_ms']:.0f} ms")
            if st.button("STORE TO EXPERIENCE MEMORY"):
                sample = st.session_state.experience_memory.store(
                    image=result.annotated_image, camera_source="streamlit",
                    detections=[d.to_dict() for d in result.detections],
                    free_space_ratio=result.scene.estimated_free_space_ratio,
                    risk_score=result.risk.score, risk_level=result.risk.level, decision=result.decision.action,
                    uncertainty_overall=result.uncertainty.overall if result.uncertainty else None,
                    capture_reason="MANUAL", source_type="STREAMLIT", model_name="classical-cv-baseline")
                st.success(f"Stored {sample.experience_id}" if sample else "Skipped (duplicate/filter)")

with tabs[1]:
    if result is None: st.info("No frame.")
    else: st.image(bgr_to_rgb(result.path_overlay if show_path else result.annotated_image), use_container_width=True)

with tabs[2]:
    st.markdown("### LIVE INPUT MANAGER")
    st.caption("ONE perception pipeline · MANY sources. Webpage URLs ≠ media streams.")
    mgr = st.session_state.input_manager
    src_label = st.selectbox("SOURCE", ["Smartphone", "Webcam", "IP Camera", "RTSP", "HTTP / MJPEG", "YouTube Live", "Twitch", "Generic Stream", "Video File"], key="uvi_src")
    mapping = {"Smartphone": SourceType.SMARTPHONE, "Webcam": SourceType.WEBCAM, "IP Camera": SourceType.IP_CAMERA,
               "RTSP": SourceType.RTSP, "HTTP / MJPEG": SourceType.HTTP_MJPEG, "YouTube Live": SourceType.YOUTUBE_LIVE,
               "Twitch": SourceType.TWITCH, "Generic Stream": SourceType.GENERIC_STREAM, "Video File": SourceType.VIDEO_FILE}
    stype = mapping[src_label]
    ident = st.text_input("URL / PATH / DEVICE", value="0" if stype == SourceType.WEBCAM else "", key="uvi_id")
    if stype in (SourceType.YOUTUBE_LIVE, SourceType.TWITCH):
        st.info("Not a direct stream. Optional yt-dlp may resolve; otherwise SOURCE NOT DIRECTLY COMPATIBLE.")
    c1,c2,c3 = st.columns(3)
    if c1.button("TEST CONNECTION", key="uvi_t"):
        st.session_state.last_diag = mgr.test_connection(InputDescriptor(stype, ident))
    if c2.button("CONNECT", key="uvi_c"):
        d = mgr.connect(InputDescriptor(stype, ident)); st.session_state.last_diag = d
        (st.success if d.connection == "ONLINE" else st.error)(d.message)
    if c3.button("DISCONNECT", key="uvi_d"):
        mgr.disconnect(); st.info("Disconnected")
    diag = st.session_state.get("last_diag") or mgr.last_diagnostics
    st.markdown(f"**STATUS** · `{diag.connection}` · Decoder `{diag.decoder}`")
    st.caption(diag.message)
    if diag.resolution: st.write(f"Resolution {diag.resolution} · FPS {diag.measured_fps} · Probe latency {diag.latency_ms} ms")
    if diag.masked_url: st.caption(f"Source: {mask_url(diag.masked_url)}")
    unc_th = st.slider("Smart capture uncertainty threshold", 0.0, 1.0, 0.35, 0.05, key="uvi_u")
    cool = st.slider("Capture cooldown (s)", 0.0, 30.0, 5.0, 0.5, key="uvi_cd")
    st.session_state.smart_policy = SmartCapturePolicy(uncertainty_threshold=unc_th, cooldown_s=cool)

with tabs[3]:
    if result is None: st.info("No analysis.")
    else:
        st.image(bgr_to_rgb(result.annotated_image), use_container_width=True)
        if result.detections: st.dataframe([d.to_dict() for d in result.detections], use_container_width=True)

with tabs[4]:
    if result is None: st.info("No analysis.")
    else:
        s = result.scene
        a,b,c,d = st.columns(4)
        a.metric("Objects", s.object_count); b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%"); d.metric("Density", f"{s.obstacle_density*100:.1f}%")

with tabs[5]:
    if result is None: st.info("No analysis.")
    elif show_path: st.image(bgr_to_rgb(result.path_overlay), use_container_width=True)

with tabs[6]:
    if result is None: st.info("No analysis.")
    else:
        st.metric("ACTION", result.decision.action); st.markdown(result.decision.reason)

with tabs[7]:
    st.markdown("### EXPERIENCE & HUMAN REVIEW")
    mem = st.session_state.experience_memory
    summ = mem.summary()
    a,b,c,d,e = st.columns(5)
    a.metric("Total", summ["total"]); b.metric("Pending", summ["pending"])
    c.metric("Accepted", summ["accepted"]); d.metric("Corrected", summ["corrected"])
    e.metric("Training-ready", summ["training_ready"])
    st.caption("Prediction ≠ ground truth. Only ACCEPTED/CORRECTED enter datasets. Inference never trains.")
    ranked = rank_for_review(mem.list_samples(100), limit=20)
    if ranked:
        choice = st.selectbox("Sample", [r.get("experience_id") or r.get("sample_id") for r in ranked])
        sel = next((r for r in ranked if (r.get("experience_id") or r.get("sample_id")) == choice), ranked[0])
        st.json({k: sel.get(k) for k in ("experience_id", "capture_reason", "review_status", "uncertainty_overall", "model_name")})
        c1,c2,c3 = st.columns(3)
        if c1.button("ACCEPT"): mem.set_review_status(choice, "accepted"); st.success("accepted")
        if c2.button("REJECT"): mem.set_review_status(choice, "rejected")
        if c3.button("CORRECT"): mem.set_review_status(choice, "corrected")
    else:
        st.info("Store experiences from Mission Control first.")

with tabs[8]:
    st.markdown("### DATASET LAB")
    approved = [s for s in st.session_state.experience_memory.list_samples(500) if s.get("review_status") in ("accepted", "corrected")]
    st.metric("Approved / training-ready", len(approved))
    if st.button("Build dataset version"):
        try:
            man = DatasetBuilder().build_from_experiences(approved)
            st.json(man.to_dict())
            for w in inspect_manifest(man.to_dict()):
                st.warning(w)
        except Exception as e:
            st.error(str(e))
    st.dataframe(DatasetBuilder().list_datasets(), use_container_width=True)

with tabs[9]:
    st.warning("Config only — does NOT train. Metrics remain NOT MEASURED. Active model is never auto-replaced.")
    if st.button("Save training config"):
        cfg = TrainingConfig(experiment_id=f"exp_{uuid.uuid4().hex[:8]}", model_name="ARQTECH",
                             training_mode="FROM_SCRATCH", dataset_id="none")
        st.success(str(save_training_config(cfg)))

with tabs[10]:
    st.warning("ARQTECH SCAFFOLD — not trained. No fabricated mAP.")
    st.json(describe_architecture())
    st.dataframe(ModelRegistry().list_models(), use_container_width=True)
    if st.button("GENERATE LEARNING REPORT"):
        rep = LearningReportGenerator().generate(experience_samples=st.session_state.experience_memory.list_samples(200))
        st.json(rep)

with tabs[11]:
    if result is None: st.info("No analysis.")
    else: st.json(result.metrics())

with tabs[12]:
    import platform
    st.markdown(f"""
| Component | Status |
|-----------|--------|
| Classical Detector | ACTIVE |
| Universal Video Input | IMPLEMENTED |
| Experience Memory | IMPLEMENTED |
| Human Review / Dataset | IMPLEMENTED |
| Training config | IMPLEMENTED (not executed) |
| ARQTECH | SCAFFOLD |
| Python | {platform.python_version()} |
""")
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")
