from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests

from .secrets import SecretProvider


@dataclass
class GroqConfig:
    model: str = "qwen/qwen3.6-27b"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    timeout_s: float = 20.0
    retries: int = 1
    max_completion_tokens: int = 512
    temperature: float = 0.2
    max_image_bytes: int = 19_000_000

    @classmethod
    def from_environment(cls) -> "GroqConfig":
        return cls(
            model=os.environ.get("GROQ_MODEL", cls.model),
            base_url=os.environ.get("GROQ_BASE_URL", cls.base_url),
            timeout_s=float(os.environ.get("GROQ_TIMEOUT_S", cls.timeout_s)),
            retries=max(0, int(os.environ.get("GROQ_RETRIES", cls.retries))),
            max_completion_tokens=max(1, int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", cls.max_completion_tokens))),
            temperature=float(os.environ.get("GROQ_TEMPERATURE", cls.temperature)),
        )

    def to_public_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroqAnalysis:
    status: str
    enabled: bool
    model: str
    latency_ms: Optional[float] = None
    parsed: Optional[dict] = None
    text: Optional[str] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class GroqClient:
    """Optional external multimodal analysis layer; never a control or label authority."""

    def __init__(self, secret_provider: Optional[SecretProvider] = None,
                 config: Optional[GroqConfig] = None, session: Optional[requests.Session] = None):
        self.secret_provider = secret_provider or SecretProvider()
        self.config = config or GroqConfig.from_environment()
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return self.secret_provider.configured()

    @property
    def status(self) -> str:
        return "READY" if self.enabled else "DISABLED"

    def public_status(self) -> dict:
        return {
            "provider": "GROQ",
            "status": self.status,
            "role": "EXTERNAL MULTIMODAL ANALYSIS LAYER",
            "model": self.config.model,
            "api_key": "CONFIGURED" if self.enabled else "NOT CONFIGURED",
            "notes": [
                "Groq is not ARQTECH.",
                "Groq is not YOLO.",
                "Groq output is not ground truth and does not control navigation.",
            ],
        }

    def analyze_image(self, image: Any, prompt: str, json_mode: bool = True) -> GroqAnalysis:
        if not self.enabled:
            return GroqAnalysis(
                status="DISABLED", enabled=False, model=self.config.model,
                notes=["GROQ_API_KEY is not configured in Streamlit Secrets or the environment."],
            )
        try:
            data_url = self._image_data_url(image)
        except Exception as exc:
            return GroqAnalysis(
                status="FAILED", enabled=True, model=self.config.model,
                error=f"image encoding failed: {type(exc).__name__}",
            )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.secret_provider.get()}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        last_error = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self.session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_s,
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.config.retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                if response.status_code >= 400:
                    return GroqAnalysis(
                        status="FAILED", enabled=True, model=self.config.model,
                        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                        error=f"HTTP {response.status_code}",
                    )
                body = response.json()
                text = self._extract_text(body)
                parsed = None
                if json_mode and text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = None
                return GroqAnalysis(
                    status="SUCCESS", enabled=True, model=self.config.model,
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    parsed=parsed, text=text,
                    request_id=response.headers.get("x-request-id") or body.get("id"),
                )
            except requests.Timeout:
                last_error = "timeout"
            except requests.RequestException as exc:
                last_error = type(exc).__name__
            except (ValueError, TypeError, KeyError):
                last_error = "invalid_response"
            if attempt < self.config.retries:
                time.sleep(0.25 * (2 ** attempt))
        return GroqAnalysis(
            status="FAILED", enabled=True, model=self.config.model,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error=last_error or "request_failed",
        )

    def _image_data_url(self, image: Any) -> str:
        if isinstance(image, (bytes, bytearray)):
            raw = bytes(image)
        elif isinstance(image, np.ndarray):
            frame = image
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[1] > 1600:
                scale = 1600 / frame.shape[1]
                frame = cv2.resize(frame, (1600, max(1, int(frame.shape[0] * scale))))
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise ValueError("could not encode image")
            raw = encoded.tobytes()
        else:
            raise TypeError("image must be bytes or numpy.ndarray")
        if len(raw) > self.config.max_image_bytes:
            raise ValueError("encoded image exceeds configured size limit")
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _extract_text(body: dict) -> str:
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)
