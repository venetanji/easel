"""Stable video IDs backed by ComfyUI's own queue and history."""
from __future__ import annotations

import base64
import re
import time

from .video_sizing import MAX_DIMENSION, MIN_DIMENSION, dimension_multiple

VIDEO_TTL_SECONDS = 24 * 60 * 60


def validate_frames(model: str, frames: int) -> None:
    if type(frames) is not int:
        raise ValueError("invalid video timing")
    if model == "minimax-h3" and 124 <= frames <= 362 and (frames - 5) % 17 == 0:
        return
    if model == "ltx-2.5" and 25 <= frames <= 289 and (frames - 1) % 24 == 0:
        return
    raise ValueError("invalid video timing")


def validate_dimensions(model: str, size: tuple[int, int]) -> None:
    multiple = dimension_multiple(model)
    if len(size) != 2 or any(type(dimension) is not int or
                            not MIN_DIMENSION <= dimension <= MAX_DIMENSION or dimension % multiple
                            for dimension in size):
        raise ValueError("invalid video dimensions")


def make_video_id(prompt_id: str, model: str, created_at: int | None = None, *,
                  frames: int | None = None, size: tuple[int, int] | None = None) -> str:
    created_at = int(time.time()) if created_at is None else created_at
    model_id = base64.urlsafe_b64encode(model.encode()).decode().rstrip("=").replace("_", ".")
    if frames is not None:
        validate_frames(model, frames)
        model_id += f":{frames}"
        if size is not None:
            validate_dimensions(model, size)
            model_id += f":{size[0]}x{size[1]}"
    elif size is not None:
        raise ValueError("video dimensions require timing")
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
        encoded = parts[2].split(":", 1)[0].replace(".", "_")
        model = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        raise ValueError("invalid video id")
    if not model or not re.fullmatch(r"[A-Za-z0-9-]{8,80}", parts[3]):
        raise ValueError("invalid video id")
    metadata = parts[2].split(":")[1:]
    if metadata:
        if len(metadata) not in (1, 2) or not re.fullmatch(r"[0-9]{1,3}", metadata[0]):
            raise ValueError("invalid video timing")
        validate_frames(model, int(metadata[0]))
        if len(metadata) == 2:
            if not re.fullmatch(r"[0-9]{3,4}x[0-9]{3,4}", metadata[1]):
                raise ValueError("invalid video dimensions")
            validate_dimensions(model, tuple(int(value) for value in metadata[1].split("x")))
    return created_at, model, parts[3]


def video_timing(video_id: str) -> dict:
    """Optional durable metadata; legacy IDs retain their original response shape."""
    parse_video_id(video_id)
    encoded = video_id.split("_", 3)[2]
    if ":" not in encoded:
        return {}
    metadata = encoded.split(":")[1:]
    frames = int(metadata[0])
    return {"frames": frames, "fps": 24, "duration": frames / 24,
            **({"size": metadata[1]} if len(metadata) == 2 else {})}
