"""Model table, OpenAI size parsing, and runtime settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelSpec:
    unet: str
    clip: str
    backend: Literal["flux2", "qwen_image_2_1"] = "flux2"
    vae: str | None = None
    default_steps: int | None = None


# OpenAI model id -> ComfyUI files and graph settings. Only place model names are defined.
# Each klein UNET needs its matching qwen text encoder (dims differ):
#   9b -> qwen-3-8b (12288-dim), 4b -> qwen-3-4b (7680-dim).
MODELS = {
    "flux2-9b": ModelSpec("flux-2-klein-9b-fp8.safetensors", "qwen_3_8b_fp8mixed.safetensors"),
    "flux2-4b": ModelSpec("flux-2-klein-4b-fp8.safetensors", "qwen_3_4b_fp4_flux2.safetensors"),
    "qwen-image-2.1": ModelSpec(
        "qwen_image_2.1_int8_convrot.safetensors",
        "qwen3vl_8b_int8_convrot.safetensors",
        backend="qwen_image_2_1",
        vae="qwen_image_2.1_vae_bf16.safetensors",
        default_steps=25,
    ),
}

# Flux.2 latent constraints.
_MULTIPLE = 16
_MIN_DIM = 256
_MAX_DIM = 1536


class UnknownModelError(Exception):
    """Raised when a requested model id is not in the table."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"model '{model}' not found")


def resolve_model(model: str) -> ModelSpec:
    try:
        return MODELS[model]
    except KeyError as exc:
        raise UnknownModelError(model) from exc


def _round_clamp(value: int) -> int:
    # Round half up to the nearest multiple (value > 0, so int() truncation == floor).
    snapped = int(value / _MULTIPLE + 0.5) * _MULTIPLE
    return max(_MIN_DIM, min(_MAX_DIM, snapped))


def parse_size(size: str | None) -> tuple[int, int] | None:
    """Parse an OpenAI `size` string into (width, height) snapped to Flux.2's
    16px grid and clamped to a sane range. Returns None for None/""/"auto"
    (caller decides the default or derives from the input image).

    Raises ValueError on anything malformed so the caller can return a 400.
    """
    if size is None:
        return None
    cleaned = size.strip().lower()
    if cleaned in ("", "auto"):
        return None
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid size '{size}'")
    w, h = int(parts[0]), int(parts[1])  # int() raises ValueError on non-digits
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid size '{size}'")
    return _round_clamp(w), _round_clamp(h)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    comfy_url: str
    api_key: str | None
    job_timeout: float
    max_inflight: int
    default_response_format: str
    default_steps: int
    variation_prompt: str
    prompt_separator: str

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            comfy_url=env.get("COMFY_URL_FLUX", "http://comfy-docker-tailscale-serve-1:8188"),
            api_key=_clean(env.get("EASEL_API_KEY")),
            job_timeout=float(env.get("COMFY_JOB_TIMEOUT", "600")),
            max_inflight=int(env.get("COMFY_MAX_INFLIGHT", "1")),
            default_response_format=env.get("EASEL_DEFAULT_RESPONSE_FORMAT", "b64_json"),
            default_steps=int(env.get("EASEL_DEFAULT_STEPS", "8")),
            variation_prompt=env.get(
                "EASEL_VARIATION_PROMPT", "recreate this image, same composition and style"
            ),
            prompt_separator=env.get("EASEL_PROMPT_SEPARATOR", "|||"),
        )
