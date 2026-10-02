"""Curated adapter identities, requirements, and fail-closed request parsing."""
from __future__ import annotations

import json
import math
from pathlib import Path


VIDEO_LORAS = {entry["id"]: entry for entry in json.loads(
    Path(__file__).with_name("video_loras.json").read_text()
)}
MAX_VIDEO_LORAS = 4


class VideoLoRAError(ValueError):
    def __init__(self, message: str, param: str = "loras"):
        self.param = param
        super().__init__(message)


def _strength(value, param):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not 0 <= value <= 2 or not math.isfinite(value):
        raise VideoLoRAError("LoRA strength must be a finite number between 0 and 2", param)
    return float(value)


def parse_video_loras(value: str | None, camera: str | None,
                      camera_strength: float | None) -> list[tuple[dict, float]]:
    selections = []
    if camera:
        camera_id = camera.strip().lower().replace("_", "-")
        if not camera_id.startswith("camera-"):
            camera_id = "camera-" + camera_id
        entry = VIDEO_LORAS.get(camera_id)
        if entry is None or entry["kind"] != "camera":
            raise VideoLoRAError("unknown camera LoRA", "camera_lora")
        strength = 0.8 if camera_strength is None else _strength(
            camera_strength, "camera_lora_strength")
        selections.append((entry, strength))
    elif camera_strength is not None:
        raise VideoLoRAError("camera_lora_strength requires camera_lora",
                             "camera_lora_strength")
    if value is not None:
        if len(value) > 8192:
            raise VideoLoRAError("loras JSON is too large")
        try:
            requested = json.loads(value)
        except (ValueError, RecursionError):
            raise VideoLoRAError("loras must be a JSON array of id/strength objects") from None
        if not isinstance(requested, list) or len(requested) > MAX_VIDEO_LORAS:
            raise VideoLoRAError(f"loras must be a JSON array with at most {MAX_VIDEO_LORAS} entries")
        for spec in requested:
            if not isinstance(spec, dict) or set(spec) - {"id", "strength"} or \
                    not isinstance(spec.get("id"), str):
                raise VideoLoRAError("each LoRA must have an id and optional strength")
            entry = VIDEO_LORAS.get(spec["id"])
            if entry is None:
                raise VideoLoRAError(f"unknown LoRA id: {spec['id']}")
            if not entry["supported"]:
                raise VideoLoRAError(f"{entry['id']} is not enabled for this model/profile; "
                                     "its required workflow or validation is missing")
            selections.append((entry, _strength(spec.get("strength", 1), "loras")))
    if len(selections) > MAX_VIDEO_LORAS:
        raise VideoLoRAError(f"at most {MAX_VIDEO_LORAS} LoRAs can be stacked")
    if len({entry["id"] for entry, _strength_value in selections}) != len(selections):
        raise VideoLoRAError("duplicate LoRA ids are not allowed")
    return selections


def installed_lora_names(info: dict) -> set[str]:
    field = info.get("LoraLoaderModelOnly", {}).get("input", {}).get("required", {}).get("lora_name")
    if isinstance(field, list) and field:
        if isinstance(field[0], list):
            return set(field[0])
        if len(field) > 1 and isinstance(field[1], dict) and "options" in field[1]:
            return set(field[1]["options"])
    raise ValueError("ComfyUI does not expose a LoRA loader filename list")
