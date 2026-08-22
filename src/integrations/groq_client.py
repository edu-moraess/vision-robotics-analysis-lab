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

try:  # Optional only during imports; requirements.txt declares the official SDK.
    from groq import Groq as OfficialGroq
except Exception:  # pragma: no cover - exercised when dependency is absent
    OfficialGroq = None


GROQ_CONNECTED = "CONNECTED"
GROQ_NOT_CONFIGURED = "NOT CONFIGURED"
GROQ_INVALID_KEY = "INVALID KEY"
GROQ_RATE_LIMITED = "RATE LIMITED"
GROQ_ERROR = "ERROR"
GROQ_OFFLINE = "OFFLINE"


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
    health_status: str = GROQ_NOT_CONFIGURED
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class GroqClient:
    """External multimodal review; never a controller, label authority or ARQTECH."""

    def __init__(self, secret_provider: Optional[SecretProvider] = None,
                 config: Optional[GroqConfig] = None, session: Optional[requests.Session] = None,
                 sdk_client: Any = None):
        self.secret_provider = secret_provider or SecretProvider()
        self.config = config or GroqConfig.from_environment()
        self.session = session
        self._sdk_client = sdk_client
        self.last_request_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_latency_ms: Optional[float] = None
        self.rate_limit_status: str = "UNKNOWN"
        self._health_status = GROQ_NOT_CONFIGURED

    @property
    def enabled(self) -> bool:
        return bool(self.secret_provider.configured())

    @property
    def client_initialized(self) -> bool:
        return self._sdk_client is not None or self.session is not None or OfficialGroq is not None

    @property
    def status(self) -> str:
        return self.health_check(probe=False)["status"]

    def public_status(self) -> dict:
        health = self.health_check(probe=False)
        return {
            "provider": "GROQ",
            "status": health["status"],
            "api_configured": bool(self.enabled),
            "client_initialized": bool(self.client_initialized and self.enabled),
            "model_configured": bool(self.config.model),
            "model": self.config.model,
            "last_request": self.last_request_at,
            "last_error": self.last_error,
            "rate_limit_status": self.rate_limit_status,
            "request_latency_ms": self.last_latency_ms,
            "role": "EXTERNAL MULTIMODAL REVIEW",
            "notes": [
                "Groq is not ARQTECH.",
                "Groq is not YOLO.",
                "Groq output is AI GENERATED and NOT GROUND TRUTH.",
                "Health CONNECTED means configured/client initialized unless probe=True is requested.",
            ],
        }

    def health_check(self, probe: bool = False) -> dict:
        if not self.enabled:
            self._health_status = GROQ_NOT_CONFIGURED
            return {"status": self._health_status, "probe": probe, "error": "GROQ_API_KEY is not configured."}
        if self.session is not None:
            self._health_status = GROQ_CONNECTED
            return {"status": self._health_status, "probe": probe, "transport": "HTTP TEST/COMPATIBILITY"}
        try:
            client = self._ensure_sdk_client()
            if probe and hasattr(client, "models") and hasattr(client.models, "list"):
                client.models.list()
            self._health_status = GROQ_CONNECTED
            return {"status": self._health_status, "probe": probe, "transport": "OFFICIAL GROQ SDK"}
        except Exception as exc:
            status = self._classify_exception(exc)
            self._health_status = status
            self.last_error = self._safe_error(exc)
            return {"status": status, "probe": probe, "error": self.last_error}

    def analyze_image(self, image: Any, prompt: str, json_mode: bool = True) -> GroqAnalysis:
        if not self.enabled:
            return GroqAnalysis(
                status="DISABLED", enabled=False, model=self.config.model,
                health_status=GROQ_NOT_CONFIGURED,
                notes=["GROQ_API_KEY is not configured in Streamlit Secrets or the environment."],
            )
        try:
            data_url = self._image_data_url(image)
        except Exception as exc:
            self.last_error = f"image encoding failed: {type(exc).__name__}"
            return GroqAnalysis(
                status="FAILED", enabled=True, model=self.config.model,
                health_status=GROQ_ERROR, error=self.last_error,
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
        started = time.perf_counter()
        self.last_request_at = time.time()
        last_error = None
        for attempt in range(self.config.retries + 1):
            try:
                if self.session is not None:
                    body, request_id = self._request_http(payload)
                else:
                    body, request_id = self._request_sdk(payload)
                text = self._extract_text(body)
                parsed = None
                if json_mode and text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = None
                latency = round((time.perf_counter() - started) * 1000.0, 3)
                self.last_latency_ms = latency
                self.last_error = None
                self._health_status = GROQ_CONNECTED
                self.rate_limit_status = "NOT HIT"
                return GroqAnalysis(
                    status="SUCCESS", enabled=True, model=self.config.model,
                    latency_ms=latency, parsed=parsed, text=text,
                    request_id=request_id, health_status=GROQ_CONNECTED,
                    notes=["AI GENERATED", "NOT GROUND TRUTH", "EXTERNAL MULTIMODAL REVIEW"],
                )
            except Exception as exc:
                last_error = self._safe_error(exc)
                self.last_error = last_error
                classified = self._classify_exception(exc)
                self._health_status = classified
                if classified == GROQ_RATE_LIMITED:
                    self.rate_limit_status = "RATE LIMITED"
                if attempt < self.config.retries and classified not in (GROQ_INVALID_KEY, GROQ_OFFLINE):
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                latency = round((time.perf_counter() - started) * 1000.0, 3)
                self.last_latency_ms = latency
                return GroqAnalysis(
                    status="FAILED", enabled=True, model=self.config.model,
                    latency_ms=latency, error=last_error,
                    health_status=classified,
                    notes=["Local perception pipeline continues independently."],
                )
        return GroqAnalysis(status="FAILED", enabled=True, model=self.config.model,
                            error=last_error or "request_failed", health_status=self._health_status)

    def _ensure_sdk_client(self):
        if self._sdk_client is None:
            if OfficialGroq is None:
                raise RuntimeError("groq SDK is not installed")
            self._sdk_client = OfficialGroq(
                api_key=self.secret_provider.get(),
                timeout=self.config.timeout_s,
                max_retries=0,
            )
        return self._sdk_client

    def _request_sdk(self, payload: dict) -> tuple[dict, Optional[str]]:
        client = self._ensure_sdk_client()
        kwargs = dict(payload)
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", "") if message is not None else ""
        return {"id": getattr(response, "id", None), "choices": [{"message": {"content": content}}]}, getattr(response, "id", None)

    def _request_http(self, payload: dict) -> tuple[dict, Optional[str]]:
        headers = {
            "Authorization": f"Bearer {self.secret_provider.get()}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            self.config.base_url, headers=headers, json=payload, timeout=self.config.timeout_s,
        )
        if response.status_code >= 400:
            error = requests.HTTPError(f"HTTP {response.status_code}")
            error.status_code = response.status_code
            raise error
        body = response.json()
        return body, response.headers.get("x-request-id") or body.get("id")

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

    @staticmethod
    def _classify_exception(exc: Exception) -> str:
        code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        message = str(exc).lower()
        if code == 401 or "401" in message or "invalid api key" in message or "authentication" in message:
            return GROQ_INVALID_KEY
        if code == 429 or "429" in message or "rate limit" in message:
            return GROQ_RATE_LIMITED
        if isinstance(exc, (requests.Timeout, TimeoutError)) or "timeout" in message:
            return GROQ_OFFLINE
        if isinstance(exc, requests.ConnectionError) or "connection" in message or "network" in message:
            return GROQ_OFFLINE
        return GROQ_ERROR

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = str(exc) or type(exc).__name__
        return message[:240].replace("gsk_", "[REDACTED]").replace("gsk-", "[REDACTED]-")
