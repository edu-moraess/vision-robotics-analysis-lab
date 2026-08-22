import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integrations.groq_client import GroqClient, GroqConfig


class SecretStub:
    def __init__(self, value=None):
        self.value = value
    def get(self):
        return self.value
    def configured(self):
        return bool(self.value)


class ResponseStub:
    status_code = 200
    headers = {"x-request-id": "req-test"}
    def json(self):
        return {
            "id": "chat-test",
            "choices": [{"message": {"content": '{"scene_summary":"ok","ambiguous_objects":[]}'}}],
        }


class SessionStub:
    def __init__(self):
        self.calls = []
    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return ResponseStub()


def test_groq_disabled_without_secret():
    client = GroqClient(secret_provider=SecretStub(), config=GroqConfig())
    result = client.analyze_image(np.zeros((8, 8, 3), dtype=np.uint8), "describe")
    assert result.status == "DISABLED"
    assert result.enabled is False


def test_groq_success_normalizes_json_and_keeps_secret_out_of_public_status():
    session = SessionStub()
    client = GroqClient(
        secret_provider=SecretStub("gsk-test-secret"),
        config=GroqConfig(model="qwen/qwen3.6-27b", retries=0),
        session=session,
    )
    result = client.analyze_image(np.zeros((8, 8, 3), dtype=np.uint8), "describe", json_mode=True)
    assert result.status == "SUCCESS"
    assert result.parsed["scene_summary"] == "ok"
    assert session.calls[0]["headers"]["Authorization"].endswith("gsk-test-secret")
    assert session.calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert "gsk-test-secret" not in str(client.public_status())


def test_pipeline_keeps_groq_advisory_and_disabled():
    from src.core.pipeline import AnalysisPipeline
    client = GroqClient(secret_provider=SecretStub(), config=GroqConfig())
    result = AnalysisPipeline(enable_groq=True, groq_client=client).run(
        np.zeros((32, 32, 3), dtype=np.uint8), run_planner=False,
    )
    assert result.groq_analysis["status"] == "DISABLED"
    assert result.metrics()["groq"]["status"] == "DISABLED"
    assert result.fused_obstacles == [] or isinstance(result.fused_obstacles, list)
