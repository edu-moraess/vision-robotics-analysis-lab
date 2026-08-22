"""Vision Robotics Analysis Lab — Engineering Control Room UI."""
from __future__ import annotations
import io, sys, time, uuid, tempfile, re, json
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.pipeline import AnalysisPipeline
from src.camera import (
    WebcamSource, IPCameraSource, SmartphoneCameraSource, VideoFileSource,
    obtain_video_metadata,
)
from src.learning import ExperienceMemory, FrameCache
from src.arqtech import ModelRegistry, describe_architecture, train_arqtech_v01, LifecycleRecord, ARQTECH_LIFECYCLE
from src.ml import DatasetBuilder, rank_for_review, LearningReportGenerator, TrainingConfig, save_training_config, inspect_manifest
from src.input import InputManager, InputDescriptor, SourceType, SmartCapturePolicy, mask_url
from src.vision.video_analysis import VideoAnalyzer
from src.vision.perception_config import (
    PERCEPTION_CURRENT, PERCEPTION_YOLO_BASELINE, PERCEPTION_FUSION,
    SMOOTHING_RAW, SMOOTHING_MOVING_AVERAGE, SMOOTHING_EXPONENTIAL,
)

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
    try:
        if hasattr(vs, "seek_frame"):
            return vs.seek_frame(int(index))
        if hasattr(vs, "read"):
            return vs.read()
    except Exception:
        return None
    return None

@st.cache_resource
def get_pipeline(min_area, conf, cell, tracking, perception_mode, model_path, iou_threshold,
                 device, image_size, max_detections, smoothing_enabled, smoothing_method,
                 smoothing_window, smoothing_alpha, enable_groq, arqtech_checkpoint):
    return AnalysisPipeline(
        min_area=min_area, conf_threshold=conf, cell_size=cell, max_image_side=1280,
        enable_tracking=tracking, perception_mode=perception_mode, model_path=model_path,
        iou_threshold=iou_threshold, device=device, image_size=image_size,
        max_detections=max_detections, smoothing_enabled=smoothing_enabled,
        smoothing_method=smoothing_method, smoothing_window=smoothing_window,
        smoothing_alpha=smoothing_alpha, enable_groq=enable_groq,
        arqtech_checkpoint=arqtech_checkpoint or None,
    )

def result_provenance(result):
    identity = result.model_identity or {}
    return {
        "model_name": identity.get("model", "UNKNOWN"),
        "model_version": identity.get("model_version", "UNKNOWN"),
        "model_backend": identity.get("model_type", "UNKNOWN"),
        "tracks": [t.to_dict() for t in (result.tracks or [])],
        "navigation_state": result.navigation_state,
        "events": list(result.track_events or []) + list(result.motion_events or []),
        "external_analysis": result.groq_analysis if (result.groq_analysis or {}).get("status") == "SUCCESS" else None,
        "masks": [d.to_dict() for d in (result.detections or [])],
        "geometry": [g.to_dict() for g in (result.geometries or [])],
        "motion": list(result.motion_observations or []),
        "trajectories": [t.to_dict() for t in (result.tracks or [])],
        "risk": result.risk.to_dict() if result.risk is not None else None,
        "occupancy": result.semantic_occupancy,
        "simulation": result.simulation_state,
        "notes": list(result.notes or []),
    }

def pipeline_for(tracking: bool):
    return get_pipeline(
        min_area, conf_thresh, cell_size, tracking, perception_mode, model_path,
        iou_threshold, device, image_size, max_detections, smoothing_enabled,
        smoothing_method, smoothing_window, smoothing_alpha, enable_groq,
        arqtech_checkpoint,
    )

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
    st.markdown("### PERCEPTION CONFIGURATION")
    perception_mode = st.selectbox("Perception mode", [PERCEPTION_CURRENT, PERCEPTION_YOLO_BASELINE, PERCEPTION_FUSION, "ARQTECH_EXPERIMENTAL"])
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    iou_threshold = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    min_area = st.slider("Min contour area (CURRENT only)", 20, 500, 80, 10)
    model_path = st.text_input("YOLO weights", value="yolo11n.pt", help="Used only by YOLO_BASELINE/FUSION; Ultralytics is optional.")
    arqtech_checkpoint = st.text_input("ARQTECH checkpoint (detection only)", value="", help="Experimental. Classification checkpoints are not treated as detectors.")
    device = st.selectbox("Inference device", ["auto", "cpu", "cuda:0"])
    image_size = st.selectbox("YOLO image size", [320, 416, 512, 640, 768], index=3)
    max_detections = st.slider("Max detections", 1, 300, 100, 1)
    enable_tracking = st.checkbox("Tracking", value=(mode == "Live Camera"))
    smoothing_enabled = st.checkbox("Temporal smoothing", value=False)
    smoothing_method = st.selectbox("Smoothing method", [SMOOTHING_RAW, SMOOTHING_MOVING_AVERAGE, SMOOTHING_EXPONENTIAL])
    smoothing_window = st.slider("Smoothing window", 1, 20, 5, 1)
    smoothing_alpha = st.slider("Exponential alpha", 0.05, 1.0, 0.35, 0.05)
    enable_groq = st.checkbox("Enable Groq multimodal review", value=False, help="Requires GROQ_API_KEY in Streamlit Secrets; output is advisory, not ground truth.")
    run_planner = st.checkbox("Image-space planner", True)
    cell_size = st.slider("Grid cell (px)", 8, 32, 16, 4)
    show_path = st.checkbox("Navigation path", True)
    analyze_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    st.caption(f"Active: {perception_mode} · YOLO is external baseline; ARQTECH is experimental")

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
                st.session_state.result = pipeline_for(False).run(preview, run_planner=run_planner)
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
            st.session_state._live_pipe = pipeline_for(enable_tracking)
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
                    result = st.session_state._live_pipe.run_packet(packet, run_planner=run_planner)
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
model_label = (result.model_identity.get("model", "UNKNOWN") if result else perception_mode)
model_status = "UNAVAILABLE" if result and result.model_identity.get("available") is False else "ACTIVE"
s3.markdown(f"**MODEL**  \n<span class='status-on'>● {model_label} · {model_status}</span>", unsafe_allow_html=True)
arq_status = describe_architecture().get("status", "NOT TRAINED")
s4.markdown(f"**ARQTECH**  \n<span class='status-off'>○ {arq_status}</span>", unsafe_allow_html=True)
s5.markdown("**LEARN**  \n<span class='status-off'>○ LOOP</span>", unsafe_allow_html=True)

tabs = st.tabs(["MISSION CONTROL", "LIVE", "VIDEO INPUT", "RECORDED VIDEO", "PERCEPTION", "SCENE", "NAVIGATION", "BRAIN", "REVIEW", "DATASET", "TRAINING", "ARQTECH", "DIAGNOSTICS", "SYSTEM", "BASELINE COMPARISON", "GROQ", "MOTION", "SIMULATION"])

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
                    capture_reason="MANUAL", source_type="STREAMLIT", source_identifier=result.source,
                    frame_id=result.frame_id, **result_provenance(result))
                st.success(f"Stored {sample.experience_id}" if sample else "Skipped")

with tabs[1]:
    st.markdown("### LIVE ROBOTIC PERCEPTION")
    if result is None: st.info("No frame.")
    else:
        col_v, col_s = st.columns([2, 1])
        with col_v:
            st.image(bgr_to_rgb(result.simulation_overlay if show_path and result.simulation_overlay is not None else (result.path_overlay if show_path else result.annotated_image)), use_container_width=True)
        with col_s:
            nav = result.navigation_state or {}
            st.markdown(f"**NAV** `{nav.get('status', 'N/A')}`")
            st.caption(nav.get("message", ""))
            st.metric("ACTION", result.decision.action)
            st.metric("RISK", result.risk.level)
            if result.inventory:
                st.markdown("**INVENTORY**")
                for k, v in result.inventory.items(): st.write(f"{k} × {v}")
        st.markdown(f"**MODEL:** `{result.model_identity.get('model', 'UNKNOWN')}` · `{result.model_identity.get('model_type', 'UNKNOWN')}`")
        st.markdown(f"**TRACKING:** `{len(result.tracks)}` tracks · **CALIBRATION:** `{result.calibration_status}`")
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
                        r0 = pipeline_for(True).run_packet(pkt0, run_planner=run_planner)
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
            # SAFE: never call vs.metadata() directly (AttributeError on old deploys)
            meta = obtain_video_metadata(vs, filename=up_vid.name)
            st.markdown("**VIDEO INFORMATION**")
            st.caption(f"source_class={meta.get('source_class')} · metadata_source={meta.get('metadata_source')}")
            st.json({k: meta.get(k) for k in (
                "filename", "format", "duration_s", "resolution", "source_fps",
                "frame_count", "file_size_bytes", "codec",
            )})
            fc = meta.get("frame_count")
            max_f = fc if isinstance(fc, int) and fc > 0 else 1
            frame_idx = st.slider("Seek frame", 0, max(0, max_f - 1), 0, key="vid_seek")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("SHOW FRAME", key="vid_show"):
                pkt = _video_frame(vs, frame_idx)
                if pkt is None or getattr(pkt, "image", None) is None:
                    st.error("Não foi possível ler este frame.")
                else:
                    r = pipeline_for(True).run_packet(pkt, run_planner=run_planner)
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
                        **result_provenance(st.session_state.result),
                    )
                    st.success(sample.experience_id if sample else "Skipped")
            if c3.button("FAST SCAN", key="vid_fast"):
                analyzer = VideoAnalyzer(pipeline_for(True))
                results, fids, tss, skipped = [], [], [], 0
                t0 = time.perf_counter()
                fps = meta.get("source_fps") if isinstance(meta.get("source_fps"), (int, float)) and meta.get("source_fps") else 25
                n = meta.get("frame_count") if isinstance(meta.get("frame_count"), int) else 0
                for i in range(0, max(n, 1), int(sample_every)):
                    pkt = _video_frame(vs, i)
                    if pkt is None or getattr(pkt, "image", None) is None:
                        skipped += 1
                        continue
                    r = analyzer.analyze_frame(
                        pkt.frame, run_planner=run_planner, timestamp=pkt.timestamp,
                        frame_id=pkt.frame_id, source=f"video:{meta.get('filename')}",
                    )
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
                st.image(bgr_to_rgb(st.session_state.result.simulation_overlay if show_path and st.session_state.result.simulation_overlay is not None else (st.session_state.result.path_overlay if show_path else st.session_state.result.annotated_image)), use_container_width=True)
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
        st.markdown(f"**MODEL:** `{result.model_identity.get('model', 'UNKNOWN')}` · **TYPE:** `{result.model_identity.get('model_type', 'UNKNOWN')}` · **VERSION:** `{result.model_identity.get('model_version', 'UNKNOWN')}`")
        if result.detections:
            st.markdown("#### NORMALIZED DETECTIONS")
            st.dataframe(result.enriched_detections, use_container_width=True)
        if result.tracks:
            st.markdown("#### TRACKS")
            st.dataframe([t.to_dict() for t in result.tracks], use_container_width=True)
        if result.track_events:
            st.markdown("#### TEMPORAL EVENTS")
            st.dataframe(result.track_events, use_container_width=True)
        st.markdown("#### SEGMENTATION")
        st.json(result.segmentation_report)
        st.caption("Positions, masks, perimeters and velocities are image-space. Real-world speed is disabled without valid camera calibration.")

with tabs[5]:
    if result is None: st.info("No analysis.")
    else:
        s = result.scene
        a,b,c,d = st.columns(4)
        a.metric("Objects", s.object_count); b.metric("Obstacles", s.obstacle_count)
        c.metric("Free space", f"{s.estimated_free_space_ratio*100:.1f}%"); d.metric("Density", f"{s.obstacle_density*100:.1f}%")
        st.markdown("#### SEMANTIC OCCUPANCY")
        st.json(result.semantic_occupancy)
        st.markdown("#### RISK ZONES")
        st.json(result.risk.to_dict().get("risk_zones", []))
        heat = result.trajectory_heatmap.get("array") if result.trajectory_heatmap else None
        if heat is not None and getattr(heat, "size", 0):
            st.image(heat, caption="TRAJECTORY HEATMAP — IMAGE-SPACE PROJECTION", clamp=True, use_container_width=True)

with tabs[6]:
    if result is None: st.info("No analysis.")
    elif show_path:
        st.image(bgr_to_rgb(result.simulation_overlay if result.simulation_overlay is not None else result.path_overlay), use_container_width=True)
        st.markdown("#### COST MAP")
        st.json(result.navigation_cost_map)
        st.markdown("#### NAVIGATION STATE")
        st.json(result.navigation_state)


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
        selected = mem.get(choice) or {}
        st.image(selected.get("image_path"), caption="HUMAN REVIEW IMAGE", use_container_width=True) if selected.get("image_path") else None
        st.json({
            "model_prediction": selected.get("model_prediction", selected.get("detections", [])),
            "groq_review": selected.get("external_analysis"),
            "review_status": selected.get("review_status"),
        })
        annotation_text = st.text_area(
            "Human annotation JSON",
            value=json.dumps(selected.get("human_annotation") or selected.get("detections", []), indent=2),
            key=f"annotation_{choice}", height=160,
        )
        try:
            edited_annotations = json.loads(annotation_text)
            if not isinstance(edited_annotations, list):
                raise ValueError("Annotation must be a JSON list")
            annotation_error = None
        except Exception as exc:
            edited_annotations, annotation_error = None, str(exc)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        if c1.button("ACCEPT", key=f"accept_{choice}"): mem.apply_review(choice, "ACCEPT", reviewer="streamlit")
        if c2.button("EDIT", key=f"edit_{choice}") and edited_annotations is not None: mem.apply_review(choice, "EDIT", edited_annotations, reviewer="streamlit")
        if c3.button("DELETE", key=f"delete_{choice}"): mem.apply_review(choice, "DELETE", reviewer="streamlit")
        if c4.button("ADD OBJECT", key=f"add_{choice}") and edited_annotations is not None: mem.apply_review(choice, "ADD OBJECT", edited_annotations, reviewer="streamlit")
        if c5.button("CHANGE CLASS", key=f"class_{choice}") and edited_annotations is not None: mem.apply_review(choice, "CHANGE CLASS", edited_annotations, reviewer="streamlit")
        if c6.button("REJECT", key=f"reject_{choice}"): mem.apply_review(choice, "REJECT", reviewer="streamlit")
        if annotation_error:
            st.warning(f"Invalid annotation JSON: {annotation_error}")
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
    st.markdown("### ARQTECH PYTORCH TRAINING")
    st.warning("v0.2 BOOTSTRAP: synthetic patch classification only. It does not create a production object detector.")
    train_epochs = st.slider("Bootstrap epochs", 1, 20, 3, 1)
    train_samples = st.slider("Synthetic samples", 64, 2000, 256, 64)
    if st.button("Save training config"):
        cfg = TrainingConfig(
            experiment_id=f"exp_{uuid.uuid4().hex[:8]}", model_name="ARQTECH",
            model_version="v0.2-modular", training_mode="FROM_SCRATCH",
            dataset_id="synthetic_patches", epochs=train_epochs,
            batch_size=32, input_resolution=(64, 64),
            dataset_scope="SYNTHETIC_BOOTSTRAP_ONLY",
            notes=["Configuration saved; training not started."],
        )
        st.success(str(save_training_config(cfg)))
    if st.button("RUN SYNTHETIC ARQTECH BOOTSTRAP"):
        with st.spinner("Running PyTorch bootstrap…"):
            train_result = train_arqtech_v01(
                epochs=train_epochs, n_samples=train_samples,
                out_dir="data/models/arqtech_v01",
            )
        st.json(train_result.to_dict())
        st.success("Training finished with experimental status; no production detection claim was made.")
    st.divider()
    st.markdown("### ARQTECH v0.3 — REAL OBJECT DETECTION")
    st.error("EXPERIMENTAL / NOT TRAINED / NOT AVAILABLE")
    st.json(LifecycleRecord().to_dict())
    st.caption("A reviewed real dataset, detection training, validation, benchmark and an explicit active checkpoint are required. v0.2 classification checkpoints cannot be loaded as detectors.")
    st.dataframe([{"lifecycle_status": status, "metrics": "NOT MEASURED"} for status in ARQTECH_LIFECYCLE], use_container_width=True)

with tabs[11]:
    st.markdown("### ARQTECH MODEL REGISTRY")
    architecture = describe_architecture()
    st.json(architecture)
    st.markdown("#### VERSION / SCOPE")
    st.dataframe([
        {"version": "v0.2-modular", "task": "SYNTHETIC PATCH CLASSIFICATION", "status": architecture.get("status", "NOT TRAINED"), "metrics": "bootstrap-only"},
        {"version": "v0.3-detection-experimental", "task": "REAL OBJECT DETECTION", "status": "EXPERIMENTAL / NOT TRAINED", "metrics": "NOT MEASURED"},
    ], use_container_width=True)
    st.dataframe(ModelRegistry().list_models(), use_container_width=True)
    st.caption("ARQTECH checkpoints are scoped to their recorded task and lifecycle; classification bootstrap is not object detection. YOLO remains an external baseline.")

with tabs[12]:
    if result is None: st.info("No analysis.")
    else:
        st.markdown("### PERFORMANCE TELEMETRY")
        st.json(result.telemetry)
        st.markdown("### MODEL PROVENANCE")
        st.json(result.model_identity)
        st.markdown("### PIPELINE METRICS")
        st.json(result.metrics())
        st.markdown("### NOTES")
        for note in result.notes:
            st.write(note)

with tabs[13]:
    import platform
    import resource
    model_status = result.model_identity if result else {"model": perception_mode, "available": "N/A"}
    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    buffer_status = st.session_state.input_manager.buffer_stats()
    st.markdown(f"""
| Component | Status |
|-----------|--------|
| Current Detector | AVAILABLE / FALLBACK |
| YOLO Baseline | {model_status.get('available', 'NOT SELECTED')} |
| Tracking | {('ACTIVE' if result and result.tracking_active else 'INACTIVE')} |
| Segmentation | {(result.segmentation_report.get('status') if result else 'IDLE')} |
| Motion / Trajectory | {(len(result.motion_observations) if result else 0)} observations |
| Temporal Smoothing | {(result.smoothing.get('method') if result else smoothing_method)} |
| Calibration | {(result.calibration_status if result else 'NOT CALIBRATED')} |
| ARQTECH | {arq_status} — NOT YOLO |
| Groq | {('ENABLED / ' + str((result.groq_analysis or {}).get('status', 'N/A')) if result and enable_groq else 'DISABLED')} |
| CPU/GPU | {device} / CUDA availability is reported by runtime only |
| Memory | {memory_mb:.1f} MB peak resident |
| Frame Buffer | {buffer_status} |
| Python | {platform.python_version()} |
""")
    if result:
        st.json({"telemetry": result.telemetry, "simulation": result.simulation_state})
    st.caption("YOLO is used as an external baseline and is not ARQTECH. Distances and velocities remain image-space until calibration is valid.")
    st.caption("https://github.com/edu-moraess/vision-robotics-analysis-lab")

with tabs[14]:
    st.markdown("### BASELINE COMPARISON")
    st.caption("Same preprocessed frame. This reports measured detector outputs only; it does not replace ground truth or establish which model is better.")
    if original_bgr is None or result is None:
        st.info("Analyze an image, camera frame or video frame first.")
    elif st.button("COMPARE CURRENT DETECTOR VS YOLO BASELINE"):
        try:
            comparison = pipeline_for(False).compare_models(
                original_bgr, timestamp=result.timestamp, frame_id=result.frame_id,
            )
            st.json(comparison)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

with tabs[15]:
    st.markdown("### GROQ MULTIMODAL REVIEW")
    st.caption("Groq is an external multimodal analysis layer and is not ARQTECH. It is not YOLO, ground truth, a controller or a training label source.")
    groq_client = pipeline_for(False).groq_client if enable_groq else None
    if groq_client is None:
        st.info("GROQ NOT CONFIGURED / DISABLED. Enable it in the sidebar and configure GROQ_API_KEY in Streamlit Secrets.")
    else:
        health_col, action_col = st.columns([2, 1])
        with health_col:
            st.json(groq_client.public_status())
        with action_col:
            if st.button("GROQ HEALTH CHECK", key="groq_health"):
                st.json(groq_client.health_check(probe=True))
        if result is not None:
            groq = result.groq_analysis or {}
            st.markdown("#### GROQ INTERPRETATION")
            st.json(groq)
            st.caption("AI GENERATED · NOT GROUND TRUTH · EXTERNAL MULTIMODAL REVIEW")
            st.markdown("#### DETECTOR OUTPUT")
            st.dataframe([d.to_dict() for d in result.detections], use_container_width=True)


with tabs[16]:
    st.markdown("### MOTION / TRAJECTORY")
    st.caption("Deterministic image-space motion. Constant velocity is a baseline, not AI prediction.")
    if result is None:
        st.info("Analyze an image, camera frame or video frame first.")
    else:
        if result.motion_observations:
            st.dataframe(result.motion_observations, use_container_width=True)
        else:
            st.info("No confirmed temporal tracks in this frame.")
        st.json({k: v for k, v in result.trajectory_heatmap.items() if k != "array"})
        heat = result.trajectory_heatmap.get("array") if result.trajectory_heatmap else None
        if heat is not None and getattr(heat, "size", 0):
            st.image(heat, caption="TRAJECTORY HEATMAP — NOT A PHYSICAL MAP", clamp=True, use_container_width=True)

with tabs[17]:
    st.markdown("### ROBOT SIMULATION")
    st.caption("SIMULATION ONLY — no physical robot control, actuator output or metric dynamics.")
    if result is None:
        st.info("Analyze an image, camera frame or video frame first.")
    else:
        if result.simulation_overlay is not None:
            st.image(bgr_to_rgb(result.simulation_overlay), caption="SIMULATION", use_container_width=True)
        st.json(result.simulation_state)
