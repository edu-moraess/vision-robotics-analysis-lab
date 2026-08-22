"""Stream resolver — webpage URLs are never passed as media streams to OpenCV."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
from .types import SourceType, InputDescriptor
from .security import mask_url

@dataclass
class ResolveResult:
    ok: bool
    source_type: SourceType
    openable: Optional[str]
    message: str
    requires_platform_adapter: bool = False
    masked_display: str = ""

_YT = re.compile(r"(youtube\.com|youtu\.be)", re.I)
_TW = re.compile(r"twitch\.tv", re.I)
_RTSP = re.compile(r"^rtsp://", re.I)
_HTTP = re.compile(r"^https?://", re.I)

class StreamResolver:
    def resolve(self, descriptor: InputDescriptor) -> ResolveResult:
        t = descriptor.source_type
        ident = (descriptor.identifier or "").strip()
        masked = mask_url(ident)
        if t == SourceType.WEBCAM:
            if not ident: ident = "0"
            if not ident.isdigit():
                return ResolveResult(False, t, None, "Webcam index must be integer", masked_display=masked)
            return ResolveResult(True, t, ident, "webcam device", masked_display=f"webcam:{ident}")
        if t == SourceType.VIDEO_FILE:
            from pathlib import Path
            if not ident or not Path(ident).exists():
                return ResolveResult(False, t, None, f"File not found: {ident}", masked_display=ident)
            return ResolveResult(True, t, ident, "local video file", masked_display=ident)
        if t in (SourceType.YOUTUBE_LIVE, SourceType.TWITCH) or _YT.search(ident) or _TW.search(ident):
            direct = self._try_ytdlp(ident)
            st = SourceType.YOUTUBE_LIVE if _YT.search(ident) else SourceType.TWITCH
            if direct:
                return ResolveResult(True, st, direct, "Resolved via yt-dlp", True, masked)
            return ResolveResult(False, st, None,
                "Platform/source not directly supported. Webpage URL is not a video stream. "
                "Install yt-dlp optionally, or provide RTSP/HTTP/MJPEG URL.", True, masked)
        if t in (SourceType.RTSP, SourceType.HTTP_MJPEG, SourceType.IP_CAMERA, SourceType.SMARTPHONE, SourceType.GENERIC_STREAM):
            if not ident:
                return ResolveResult(False, t, None, "Stream URL required", masked_display="")
            if not (_RTSP.match(ident) or _HTTP.match(ident)):
                return ResolveResult(False, t, None, "URL must be rtsp:// or http(s)://", masked_display=masked)
            return ResolveResult(True, t, ident, "direct network stream", masked_display=masked)
        if _RTSP.match(ident):
            return ResolveResult(True, SourceType.RTSP, ident, "auto RTSP", masked_display=masked)
        if _HTTP.match(ident):
            return ResolveResult(True, SourceType.HTTP_MJPEG, ident, "auto HTTP", masked_display=masked)
        return ResolveResult(False, SourceType.UNKNOWN, None, "Unable to resolve source", masked_display=masked)

    def _try_ytdlp(self, page_url: str):
        try:
            import yt_dlp
        except ImportError:
            return None
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "best[height<=720]/best"}) as ydl:
                info = ydl.extract_info(page_url, download=False)
                if info and info.get("url"):
                    return str(info["url"])
                for f in reversed(info.get("formats") or []):
                    if f.get("url") and f.get("vcodec") not in (None, "none"):
                        return str(f["url"])
        except Exception:
            return None
        return None
