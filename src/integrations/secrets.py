from __future__ import annotations

import os
from typing import Optional


class SecretProvider:
    """Read runtime secrets without writing them to project files or logs."""

    def __init__(self, key_name: str = "GROQ_API_KEY"):
        self.key_name = key_name

    def get(self) -> Optional[str]:
        value = self._from_streamlit()
        if value:
            return value
        value = os.environ.get(self.key_name)
        return value.strip() if value else None

    def configured(self) -> bool:
        return bool(self.get())

    def status(self) -> str:
        return "CONFIGURED" if self.configured() else "DISABLED"

    def _from_streamlit(self) -> Optional[str]:
        try:
            import streamlit as st
            if self.key_name not in st.secrets:
                return None
            value = st.secrets[self.key_name]
            return str(value).strip() if value else None
        except Exception:
            return None

    @staticmethod
    def redact(value: str, visible: int = 4) -> str:
        if not value:
            return ""
        if len(value) <= visible:
            return "*" * len(value)
        return value[:visible] + "*" * max(4, len(value) - visible)
