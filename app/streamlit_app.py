"""Vision Robotics Analysis Lab — Engineering Control Room UI."""
from __future__ import annotations
import io, sys, time, uuid, tempfile, re
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
from src.vision.video_analysis import VideoAnalyzer

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

def _video_frame(vs, index: int = 0):
    """Safe frame read — works even if seek_frame is missing on older deploys."""
    try:
        if hasattr(vs, "seek_frame"):
            return vs.seek_frame(int(index))
        if hasattr(vs, "read"):
            return vs.read()
    except Exception:
        return None
    return None

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
    st.caption("Active: Classical CV · ONE pipeline for image / video / live")

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

tabs = st.tabs(["MISSION CONTROL", "LIVE", "VIDEO INPUT", "RECORDED VIDEO", "PERCEPTION", "SCENE", "NAVIGATION", "BRAIN", "REVIEW", "DATASET", "TRAINING", "ARQTECH", "DIAGNOSTICS", "SYSTEM"])

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
                st.success(f"Stored {sample.experience_id}" if sample else "Skipped")

with tabs[1]:
    st.markdown("### LIVE ROBOTIC PERCEPTION")
    if result is None: st.info("No frame.")
    else:
        col_v, col_s = st.columns([2, 1])
        with col_v:
            st.image(bgr_to_rgb(result.path_overlay if show_path else result.annotated_image), use_container_width=True)
        with col_s:
            nav = result.navigation_state or {}
            st.markdown(f"**NAV** `{nav.get('status', 'N/A')}`")
            st.caption(nav.get("message", ""))
            st.metric("ACTION", result.decision.action)
            st.metric("RISK", result.risk.level)
            if result.inventory:
                st.markdown("**INVENTORY**")
                for k, v in result.inventory.items(): st.write(f"{k} × {v}")
        for line in (result.narrative or []):
            st.write("• " + line)

with tabs[2]:
    st.markdown("### LIVE INPUT MANAGER")
    mgr = st.session_state.input_manager
    src_label = st.selectbox("SOURCE", ["Smartphone", "Webcam", "IP Camera", "RTSP", "HTTP / MJPEG", "YouTube Live", "Twitch", "Generic Stream", "Video File"], key="uvi_src")
    mapping = {"Smartphone": SourceType.SMARTPHONE, "Webcam": SourceType.WEBCAM, "IP Camera": SourceType.IP_CAMERA,
               "RTSP": SourceType.RTSP, "HTTP / MJPEG": SourceType.HTTP_MJPEG, "YouTube Live": SourceType.YOUTUBE_LIVE,
               "Twitch": SourceType.TWITCH, "Generic Stream": SourceType.GENERIC_STREAM, "Video File": SourceType.VIDEO_FILE}
    stype = mapping[src_label]
    ident = st.text_input("URL / PATH / DEVICE", value="0" if stype == SourceType.WEBCAM else "", key="uvi_id")
    c1,c2,c3 = st.columns(3)
    if c1.button("TEST CONNECTION", key="uvi_t"):
        st.session_state.last_diag = mgr.test_connection(InputDescriptor(stype, ident))
    if c2.button("CONNECT", key="uvi_c"):
        d = mgr.connect(InputDescriptor(stype, ident)); st.session_state.last_diag = d
        (st.success if d.connection == "ONLINE" else st.error)(d.message)
    if c3.button("DISCONNECT", key="uvi_d"):
        mgr.disconnect()
    diag = st.session_state.get("last_diag") or mgr.last_diagnostics
    st.caption(f"{diag.connection} · {diag.decoder} · {diag.message}")

with tabs[3]:
    st.markdown("### RECORDED VIDEO LAB")
    st.caption("Mesmo pipeline do live. Preferir MP4 H.264.")
    up_vid = st.file_uploader("Upload vídeo gravado", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"], key="vid_up")
    sample_every = st.selectbox("Sample every N frames", [1, 2, 5, 10], index=1, key="vid_sample")
    if up_vid is not None:
        safe = re.sub(r"[^\w.\-]+", "_", up_vid.name)
        tmp_dir = Path(tempfile.gettempdir()) / "vral_videos"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{up_vid.size}_{safe}"
        if not tmp.exists() or tmp.stat().st_size != up_vid.size:
            tmp.write_bytes(up_vid.getbuffer())
        st.write(f"**Ficheiro:** `{up_vid.name}` · {up_vid.size} bytes")
        need_open = (
            st.session_state.get("video_src") is None
            or st.session_state.get("video_path") != str(tmp)
            or not getattr(st.session_state.get("video_src"), "is_available", lambda: False)()
        )
        if need_open:
            if st.session_state.get("video_src") is not None:
                try: st.session_state.video_src.stop()
                except Exception: pass
            vs = VideoFileSource(str(tmp), loop=False)
            st_status = vs.start()
            st.session_state.video_src = vs
            st.session_state.video_path = str(tmp)
            st.session_state.video_report = None
            if not st_status.online:
                st.error(st_status.message)
                st.info("Dica: exportar como MP4 H.264 ou usar frames JPG.")
            else:
                st.success("Vídeo aberto.")
                pkt0 = _video_frame(vs, 0)
                if pkt0 is not None and getattr(pkt0, "image", None) is not None:
                    try:
                        r0 = get_pipeline(min_area, conf_thresh, cell_size, True).run(pkt0.image, run_planner=run_planner)
                        st.session_state.result = r0
                        st.session_state.original_bgr = pkt0.image
                        st.session_state._last_vid_pkt = pkt0
                    except Exception as ex:
                        st.warning(f"1º frame aberto, análise falhou: {ex}")
                else:
                    st.warning("Vídeo aberto, mas não foi possível ler o 1º frame. Use SHOW FRAME.")
        vs = st.session_state.get("video_src")
        if vs is None or not vs.is_available():
            st.warning("Fonte de vídeo indisponível.")
        else:
            meta = vs.metadata() if hasattr(vs, "metadata") else {}
            st.json({k: meta.get(k) for k in ("filename", "format", "duration_s", "resolution", "source_fps", "frame_count", "file_size_bytes")})
            max_f = meta.get("frame_count") if isinstance(meta.get("frame_count"), int) and meta.get("frame_count") > 0 else 1
            frame_idx = st.slider("Seek frame", 0, max(0, max_f - 1), 0, key="vid_seek")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("SHOW FRAME", key="vid_show"):
                pkt = _video_frame(vs, frame_idx)
                if pkt is None or getattr(pkt, "image", None) is None:
                    st.error("Não foi possível ler este frame.")
                else:
                    r = get_pipeline(min_area, conf_thresh, cell_size, True).run(pkt.image, run_planner=run_planner)
                    st.session_state.result = r
                    st.session_state.original_bgr = pkt.image
                    st.session_state._last_vid_pkt = pkt
            if c2.button("CAPTURE EXPERIENCE", key="vid_cap"):
                if st.session_state.result is None:
                    st.warning("Analise um frame primeiro.")
                else:
                    pkt = st.session_state.get("_last_vid_pkt")
                    sample = st.session_state.experience_memory.store(
                        image=st.session_state.result.annotated_image,
                        camera_source=f"video:{meta.get('filename')}",
                        detections=[d.to_dict() for d in st.session_state.result.detections],
                        free_space_ratio=st.session_state.result.scene.estimated_free_space_ratio,
                        risk_score=st.session_state.result.risk.score,
                        risk_level=st.session_state.result.risk.level,
                        decision=st.session_state.result.decision.action,
                        uncertainty_overall=st.session_state.result.uncertainty.overall if st.session_state.result.uncertainty else None,
                        capture_reason="MANUAL", source_type="RECORDED_VIDEO",
                        source_identifier=str(meta.get("filename")),
                        frame_id=pkt.frame_id if pkt else frame_idx,
                        model_name="classical-cv-baseline",
                    )
                    st.success(sample.experience_id if sample else "Skipped")
            if c3.button("FAST SCAN", key="vid_fast"):
                analyzer = VideoAnalyzer(get_pipeline(min_area, conf_thresh, cell_size, True))
                results, fids, tss, skipped = [], [], [], 0
                t0 = time.perf_counter()
                fps = meta.get("source_fps") if isinstance(meta.get("source_fps"), (int, float)) and meta.get("source_fps") else 25
                n = meta.get("frame_count") if isinstance(meta.get("frame_count"), int) else 0
                for i in range(0, max(n, 1), int(sample_every)):
                    pkt = _video_frame(vs, i)
                    if pkt is None or getattr(pkt, "image", None) is None:
                        skipped += 1
                        continue
                    r = analyzer.analyze_frame(pkt.image, run_planner=run_planner)
                    results.append(r); fids.append(getattr(pkt, "frame_id", i)); tss.append(i / float(fps) if fps else float(i))
                if results:
                    st.session_state.result = results[-1]
                    st.session_state.original_bgr = results[-1].annotated_image
                    st.session_state.video_report = analyzer.build_report(str(meta.get("filename")), results, fids, tss, skipped, time.perf_counter() - t0).to_dict()
                    st.success(f"Analisados {len(results)} frames")
                else:
                    st.error("Nenhum frame analisado.")
            if c4.button("STOP VIDEO", key="vid_stop"):
                try: vs.stop()
                except Exception: pass
                st.session_state.video_src = None
                st.session_state.video_path = None
            if st.session_state.result is not None and st.session_state.get("video_path") == str(tmp):
                st.image(bgr_to_rgb(st.session_state.result.path_overlay if show_path else st.session_state.result.annotated_image), use_container_width=True)
                for line in (st.session_state.result.narrative or [])[:6]:
                    st.write("• " + line)
            if st.session_state.get("video_report"):
                st.markdown("**VIDEO REPORT**")
                st.json(st.session_state.video_report)
    else:
        st.info("Carregue MP4/AVI/MOV/MKV/WebM (preferir H.264).")

with tabs[4]:
    if result is None: st.info("No analysis.")
    else:
        st.image(bgr_to_rgb(result.annotated_image), use_container_width=True)
        if result.detections: st.dataframe([d.to_dict() for d in result.detections], use_container_width=True)

with tabs[5]:
    if result is None: st.info("No analysis.")
    else:
        s = result.scene
        a,b,c,d = st.columns(4)
        a.metric("Objects", s.object_count); b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%"); d.metric("Density", f"{s.obstacle_density*100:.1f}%")

with tabs[6]:
    if result is None: st.info("No analysis.")
    elif show_path: st.image(bgr_to_rgb(result.path_overlay), use_container_width=True)

with tabs[7]:
    if result is None: st.info("No analysis.")
    else: st.metric("ACTION", result.decision.action); st.markdown(result.decision.reason)

with tabs[8]:
    st.markdown("### EXPERIENCE & HUMAN REVIEW")
    mem = st.session_state.experience_memory
    summ = mem.summary()
    a,b,c,d,e = st.columns(5)
    a.metric("Total", summ["total"]); b.metric("Pending", summ["pending"])
    c.metric("Accepted", summ["accepted"]); d.metric("Corrected", summ["corrected"])
    e.metric("Training-ready", summ["training_ready"])
    ranked = rank_for_review(mem.list_samples(100), limit=20)
    if ranked:
        choice = st.selectbox("Sample", [r.get("experience_id") or r.get("sample_id") for r in ranked])
        c1,c2,c3 = st.columns(3)
        if c1.button("ACCEPT"): mem.set_review_status(choice, "accepted")
        if c2.button("REJECT"): mem.set_review_status(choice, "rejected")
        if c3.button("CORRECT"): mem.set_review_status(choice, "corrected")
    else: st.info("Store experiences first.")

with tabs[9]:
    st.markdown("### DATASET LAB")
    approved = [s for s in st.session_state.experience_memory.list_samples(500) if s.get("review_status") in ("accepted", "corrected")]
    st.metric("Approved", len(approved))
    if st.button("Build dataset version"):
        try:
            man = DatasetBuilder().build_from_experiences(approved)
            st.json(man.to_dict())
            for w in inspect_manifest(man.to_dict()): st.warning(w)
        except Exception as e: st.error(str(e))

with tabs[10]:
    st.warning("Config only — does NOT train.")
    if st.button("Save training config"):
        cfg = TrainingConfig(experiment_id=f"exp_{uuid.uuid4().hex[:8]}", model_name="ARQTECH",
                             training_mode="FROM_SCRATCH", dataset_id="none")
        st.success(str(save_training_config(cfg)))

with tabs[11]:
    st.warning("ARQTECH SCAFFOLD — not trained.")
    st.json(describe_architecture())
    st.dataframe(ModelRegistry().list_models(), use_container_width=True)

with tabs[12]:
    if result is None: st.info("No analysis.")
    else: st.json(result.metrics())

with tabs[13]:
    import platform
    st.markdown(f"""
| Component | Status |
|-----------|--------|
| Classical Detector | ACTIVE |
| Recorded Video Lab | FIXED (safe seek) |
| ARQTECH | SCAFFOLD |
| Python | {platform.python_version()} |
""")
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")
