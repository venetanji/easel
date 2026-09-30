"""Stable video IDs backed by ComfyUI's own queue and history."""
from __future__ import annotations

import base64
import re
import time

VIDEO_TTL_SECONDS = 24 * 60 * 60


def make_video_id(prompt_id: str, model: str, created_at: int | None = None) -> str:
    created_at = int(time.time()) if created_at is None else created_at
    model_id = base64.urlsafe_b64encode(model.encode()).decode().rstrip("=").replace("_", ".")
    return f"video_{created_at}_{model_id}_{prompt_id}"


def parse_video_id(video_id: str) -> tuple[int, str, str]:
    parts = video_id.split("_", 3)
    if len(parts) != 4 or parts[0] != "video" or not parts[1].isdigit():
        raise ValueError("invalid video id")
    created_at = int(parts[1])
    now = int(time.time())
    if created_at > now + 60:
        raise ValueError("invalid video id")
    if created_at + VIDEO_TTL_SECONDS <= now:
        raise ValueError("video id has expired")
    try:
        encoded = parts[2].replace(".", "_")
        model = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        raise ValueError("invalid video id")
    if not model or not re.fullmatch(r"[A-Za-z0-9-]{8,80}", parts[3]):
        raise ValueError("invalid video id")
    return created_at, model, parts[3]
