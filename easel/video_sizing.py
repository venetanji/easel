"""Shared video presets and bounded, model-aligned resolution/duration admission."""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MAX_PIXELS = 2_097_152
DEFAULT_MAX_PIXEL_FRAMES = 160_000_000
MIN_DIMENSION = 256
MAX_DIMENSION = 2048
FPS = 24

VIDEO_SIZES = {
    "512x320": (512, 320), "320x512": (320, 512),
    "640x384": (640, 384), "384x640": (384, 640),
    "768x512": (768, 512), "512x768": (512, 768),
    "1024x576": (1024, 576), "576x1024": (576, 1024),
    "1280x704": (1280, 704), "704x1280": (704, 1280),
    "1792x1024": (1792, 1024), "1024x1792": (1024, 1792),
    "1920x1088": (1920, 1088), "1088x1920": (1088, 1920),
    "512x512": (512, 512), "640x640": (640, 640),
    "768x768": (768, 768), "1024x1024": (1024, 1024),
    "1280x1280": (1280, 1280),
}
SIZE_ALIASES = {"1280x720": "1280x704", "720x1280": "704x1280"}
VIDEO_SIZES.update({alias: VIDEO_SIZES[target] for alias, target in SIZE_ALIASES.items()})


class VideoSizingError(ValueError):
    def __init__(self, message: str, *, param: str = "size", code: str | None = None,
                 details: dict | None = None):
        self.param = param
        self.code = code
        self.details = details or {}
        super().__init__(message)


def dimension_multiple(model: str) -> int:
    if model == "ltx-2.5":
        return 64
    if model == "minimax-h3":
        return 32
    raise ValueError(f"unknown video model '{model}'")


@dataclass(frozen=True)
class VideoLimits:
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_pixel_frames: int = DEFAULT_MAX_PIXEL_FRAMES

    def __post_init__(self):
        for name in ("max_pixels", "max_pixel_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"video {name} must be a positive integer")

    def max_frames(self, model: str, width: int, height: int) -> int:
        if width * height > self.max_pixels:
            return 0
        maximum = self.max_pixel_frames // (width * height)
        if model == "ltx-2.5":
            return ((min(maximum, 289) - 1) // 24) * 24 + 1 if maximum >= 25 else 0
        if model == "minimax-h3":
            return ((min(maximum, 362) - 5) // 17) * 17 + 5 if maximum >= 124 else 0
        raise ValueError(f"unknown video model '{model}'")

    def check_frames(self, model: str, width: int, height: int, frames: int) -> None:
        maximum = self.max_frames(model, width, height)
        if frames <= maximum:
            return
        param = "seconds" if model == "ltx-2.5" else "frames"
        limit = (maximum - 1) // FPS if maximum and param == "seconds" else maximum
        message = (f"{width}x{height} cannot fit {model}'s minimum duration under the configured video budget; reduce size"
                   if not maximum else
                   f"{width}x{height} allows at most {limit} {param} under the configured video budget; reduce size or duration")
        raise VideoSizingError(
            message,
            param=param, code="video_resource_limit",
            details={"max_frames": maximum, "max_pixel_frames": self.max_pixel_frames,
                     "requested_pixel_frames": width * height * frames,
                     **({"max_seconds": limit} if model == "ltx-2.5" else {})},
        )

    def discovery(self, model: str) -> dict:
        presets = dict(VIDEO_SIZES)
        if model == "minimax-h3":
            presets["864x480"] = (864, 480)
        entries = []
        for name, (width, height) in presets.items():
            maximum = self.max_frames(model, width, height)
            if not maximum:
                continue
            entry = {"size": name, "width": width, "height": height, "max_frames": maximum,
                     "validation": "admission_only"}
            if model == "ltx-2.5":
                entry["max_seconds"] = (maximum - 1) // FPS
            entries.append(entry)
        return {
            "sizes": [entry["size"] for entry in entries],
            "size_presets": entries,
            "size_constraints": {
                "custom_sizes": True, "format": "WIDTHxHEIGHT",
                "min_dimension": MIN_DIMENSION, "max_dimension": MAX_DIMENSION,
                "dimension_multiple": dimension_multiple(model),
                "max_pixels": self.max_pixels, "max_pixel_frames": self.max_pixel_frames,
                "budget_kind": "admission_heuristic", "oom_guarantee": False,
                "aliases": dict(SIZE_ALIASES),
            },
            "failure_recovery": {"oom_code": "upstream_out_of_memory", "automatic_retry": False,
                                 "suggested_action": "reduce_size_or_duration"},
        }


def parse_video_size(value: str | None, *, model: str = "ltx-2.5",
                     limits: VideoLimits = VideoLimits()) -> tuple[int, int]:
    default = "864x480" if model == "minimax-h3" else "1280x720"
    size = (value or default).strip().lower()
    size = SIZE_ALIASES.get(size, size)
    if not re.fullmatch(r"[0-9]{3,4}x[0-9]{3,4}", size):
        raise VideoSizingError("size must use WIDTHxHEIGHT format")
    width, height = (int(part) for part in size.split("x"))
    multiple = dimension_multiple(model)
    if any(not MIN_DIMENSION <= dimension <= MAX_DIMENSION or dimension % multiple
           for dimension in (width, height)):
        raise VideoSizingError(
            f"{model} dimensions must be multiples of {multiple} in "
            f"{MIN_DIMENSION}..{MAX_DIMENSION}; sizes are not silently rounded",
        )
    if width * height > limits.max_pixels:
        raise VideoSizingError(
            f"size exceeds the configured {limits.max_pixels}-pixel video budget",
            code="video_resource_limit", details={"max_pixels": limits.max_pixels},
        )
    return width, height
