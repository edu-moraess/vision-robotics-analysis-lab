from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class SourceType(str, Enum):
    WEBCAM = "WEBCAM"
    SMARTPHONE = "SMARTPHONE"
    IP_CAMERA = "IP_CAMERA"
    RTSP = "RTSP"
    HTTP_MJPEG = "HTTP_MJPEG"
    VIDEO_FILE = "VIDEO_FILE"
    GENERIC_STREAM = "GENERIC_STREAM"
    YOUTUBE_LIVE = "YOUTUBE_LIVE"
    TWITCH = "TWITCH"
    UNKNOWN = "UNKNOWN"

@dataclass
class InputDescriptor:
    source_type: SourceType
    identifier: str
    label: str = ""
