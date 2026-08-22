"""
Safe video metadata extraction — never crashes the app.

Works with:
- VideoFileSource.metadata() when present
- objects exposing OpenCV VideoCapture (_cap)
- duck-typed attributes
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

NA = "N/A"


def empty_metadata(filename: str = "", source_type: str = "RECORDED_VIDEO") -> Dict[str, Any]:
    return {
        "filename": filename or NA,
        "format": NA,
        "duration_s": None,
        "duration": NA,
        "fps": None,
        "source_fps": NA,
        "width": None,
        "height": None,
        "resolution": NA,
        "frame_count": None,
        "codec": NA,
        "file_size_bytes": None,
        "file_size": NA,
        "source_type": source_type,
        "path": NA,
        "decoder": NA,
        "source_class": NA,
        "metadata_source": "empty",
    }


def _from_opencv_cap(cap, path: str = "") -> Dict[str, Any]:
    meta = empty_metadata(Path(path).name if path else "", "RECORDED_VIDEO")
    meta["path"] = path or NA
    meta["metadata_source"] = "opencv_capture"
    try:
        fps = float(cap.get(5))  # CAP_PROP_FPS
        w = int(cap.get(3))
        h = int(cap.get(4))
        n = int(cap.get(7))
    except Exception as e:
        logger.debug("opencv cap props failed: %s", e)
        return meta
    meta["fps"] = fps if fps and fps > 0 else None
    meta["source_fps"] = fps if fps and fps > 0 else NA
    meta["width"] = w if w > 0 else None
    meta["height"] = h if h > 0 else None
    meta["resolution"] = (w, h) if w > 0 and h > 0 else NA
    meta["frame_count"] = n if n > 0 else None
    if fps and fps > 0 and n and n > 0:
        meta["duration_s"] = n / fps
        meta["duration"] = n / fps
    else:
        meta["duration_s"] = None
        meta["duration"] = NA
    return meta


def obtain_video_metadata(vs: Any, filename: str = "") -> Dict[str, Any]:
    """
    Robust metadata for any video source object.
    Prefer vs.metadata() when implemented.
    Fall back to OpenCV capture props.
    Never raises to the UI for expected decoder gaps.
    """
    source_class = type(vs).__name__ if vs is not None else "None"
    base = empty_metadata(filename, "RECORDED_VIDEO")
    base["source_class"] = source_class

    if vs is None:
        base["metadata_source"] = "none"
        return base

    if hasattr(vs, "metadata") and callable(getattr(vs, "metadata")):
        try:
            raw = vs.metadata()
            if isinstance(raw, dict):
                out = empty_metadata(filename or str(raw.get("filename") or ""), "RECORDED_VIDEO")
                out.update({k: v for k, v in raw.items() if v is not None})
                if out.get("source_fps") not in (None, NA) and out.get("fps") is None:
                    try:
                        out["fps"] = float(out["source_fps"])
                    except (TypeError, ValueError):
                        pass
                if isinstance(out.get("resolution"), tuple) and len(out["resolution"]) == 2:
                    out["width"], out["height"] = out["resolution"]
                if out.get("duration_s") not in (None, NA) and out.get("duration") in (None, NA):
                    out["duration"] = out["duration_s"]
                if out.get("file_size_bytes") not in (None, NA):
                    out["file_size"] = out["file_size_bytes"]
                out["source_class"] = source_class
                out["metadata_source"] = "vs.metadata()"
                out["decoder"] = getattr(vs, "_message", NA) if hasattr(vs, "_message") else NA
                return out
        except Exception as e:
            logger.warning("vs.metadata() failed (%s): %s", source_class, e)

    cap = getattr(vs, "_cap", None)
    path = getattr(vs, "path", "") or filename
    if cap is not None:
        try:
            opened = True
            if hasattr(cap, "isOpened"):
                opened = bool(cap.isOpened())
            if opened:
                m = _from_opencv_cap(cap, str(path))
                m["source_class"] = source_class
                m["decoder"] = "opencv"
                return m
        except Exception as e:
            logger.warning("opencv metadata failed: %s", e)

    if hasattr(vs, "status") and callable(vs.status):
        try:
            st = vs.status()
            res = getattr(st, "resolution", None)
            if res and isinstance(res, (tuple, list)) and len(res) == 2:
                base["width"], base["height"] = int(res[0]), int(res[1])
                base["resolution"] = (base["width"], base["height"])
            fps = getattr(st, "measured_fps", None)
            if fps:
                base["fps"] = float(fps)
                base["source_fps"] = float(fps)
            base["metadata_source"] = "status()"
        except Exception as e:
            logger.debug("status() metadata failed: %s", e)

    if path:
        p = Path(str(path))
        if p.exists():
            base["filename"] = p.name
            base["format"] = p.suffix.lower().lstrip(".") or NA
            try:
                base["file_size_bytes"] = p.stat().st_size
                base["file_size"] = base["file_size_bytes"]
            except OSError:
                pass
            base["path"] = str(p)
            base["metadata_source"] = "filesystem"

    return base
