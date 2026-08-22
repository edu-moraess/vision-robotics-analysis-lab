from __future__ import annotations
import re

def mask_url(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"(://[^:/@\s]+):([^@/\s]+)@", r"\1:******@", url)
