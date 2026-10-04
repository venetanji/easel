"""Offline H3 graph baseline; HTTP registration and media admission are separate."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .workflow_graph import WorkflowGraph

VIDEO_MODEL_ID = "minimax-h3"
FPS = 24
MIN_FRAMES = 124
MAX_FRAMES = 362
MAX_REFERENCES = 2
MAX_TEMPORAL_GUIDES = 3
MAX_GUIDE_FRAMES = 39
UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF_UNET_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL8_LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TEXT_ENCODER_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"


@dataclass(frozen=True)
class TemporalGuide:
    """Ordered managed input images forming one timeline guide, not semantic refs."""

    image_filenames: tuple[str, ...]
    frame_index: int


def _integer(name: str, value: int, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")


def _image_name(value: str) -> str:
    if (not isinstance(value, str) or len(value) > 200 or
            re.fullmatch(r"[A-Za-z0-9_-]+\.(?:png|jpg|jpeg)", value) is None):
        raise ValueError("image filename must be a managed PNG/JPEG input basename")
    return value


def _references(values, *, required: bool) -> tuple[str, ...]:
    minimum = 1 if required else 0
    if not isinstance(values, (list, tuple)) or not minimum <= len(values) <= MAX_REFERENCES:
        raise ValueError(f"reference_filenames must contain {minimum}..{MAX_REFERENCES} images")
    return tuple(_image_name(value) for value in values)


def _guides(values, frames: int) -> tuple[TemporalGuide, ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_TEMPORAL_GUIDES:
        raise ValueError(f"guides must contain 1..{MAX_TEMPORAL_GUIDES} TemporalGuide entries")
    for guide in values:
        if not isinstance(guide, TemporalGuide) or not isinstance(guide.image_filenames, tuple):
            raise ValueError("guides must be TemporalGuide entries with immutable image tuples")
        count = len(guide.image_filenames)
        if not 1 <= count <= MAX_GUIDE_FRAMES or (count != 1 and (count - 5) % 17):
            raise ValueError("each guide must have exactly 1, 5, 22, or 39 ordered frames")
        for name in guide.image_filenames:
            _image_name(name)
        _integer("guide frame_index", guide.frame_index, 0, frames - count)
    ordered = tuple(sorted(values, key=lambda guide: guide.frame_index))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.frame_index + len(previous.image_filenames) > current.frame_index:
            raise ValueError("temporal guide ranges must not overlap")
    return ordered


def _build_video(
    prompt: str, *, width: int, height: int, frames: int, seed: int,
    filename_prefix: str, reference_mode: bool = False,
    image_filename: str | None = None, reference_filenames: tuple[str, ...] = (),
    ref_image_size: str = "match", guides: tuple[TemporalGuide, ...] = (),
) -> dict:
    if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
        raise ValueError("prompt must be nonblank and contain no NUL")
    for name, value in (("width", width), ("height", height)):
        _integer(name, value, 32, 16384)
        if value % 32:
            raise ValueError(f"{name} must be a multiple of 32")
    _integer("frames", frames, MIN_FRAMES, MAX_FRAMES)
    if (frames - 5) % 17:
        raise ValueError("frames must lie on H3's 17k+5 grid (124..362)")
    _integer("seed", seed, 0, 2**64 - 1)
    if (not isinstance(filename_prefix, str) or len(filename_prefix) > 200 or
            re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", filename_prefix) is None):
        raise ValueError("filename_prefix must be a safe relative literal prefix")
    if not isinstance(ref_image_size, str) or ref_image_size not in ("match", "max"):
        raise ValueError("ref_image_size must be match or max")

    graph = WorkflowGraph()
    model = graph.node("UNETLoader", unet_name=REF_UNET_NAME if reference_mode else UNET_NAME,
                       weight_dtype="default")
    if not reference_mode:
        model = graph.node("LoraLoaderModelOnly", model=model[0], lora_name=FL8_LORA_NAME,
                           strength_model=1.0)
        model = graph.node("MiniMaxH3SigmaShift", model=model[0], shift_video=12.0, shift_audio=3.0)
    clip = graph.node("CLIPLoader", clip_name=TEXT_ENCODER_NAME, type="minimax", device="default")
    video_vae = graph.node("VAELoader", vae_name=VIDEO_VAE_NAME)
    audio_vae = graph.node("VAELoader", vae_name=AUDIO_VAE_NAME)
    inputs = dict(clip=clip[0], vae=video_vae[0], prompt=prompt,
                  width=width, height=height, length=frames)
    if reference_mode:
        inputs.update(audio_vae=audio_vae[0], ref_image_size=ref_image_size)
        for index, name in enumerate(reference_filenames):
            inputs[f"ref_images.ref_image_{index}"] = graph.node("LoadImage", image=name)[0]
        av = graph.node("MiniMaxH3ReferenceToVideo", **inputs)
    else:
        if image_filename is not None:
            inputs["first_frame"] = graph.node("LoadImage", image=image_filename)[0]
        av = graph.node("MiniMaxH3ImageToVideo", **inputs)
    conditioning = av[0]
    for guide in guides:
        images = graph.node("LoadImage", image=guide.image_filenames[0])[0]
        for name in guide.image_filenames[1:]:
            images = graph.node("ImageBatch", image1=images,
                                image2=graph.node("LoadImage", image=name)[0])[0]
        conditioning = graph.node("MiniMaxH3AddGuide", positive=conditioning, latent=av[1],
                                  vae=video_vae[0], image=images, frame_idx=guide.frame_index)[0]
    guider = graph.node("BasicGuider", model=model[0], conditioning=conditioning)
    noise = graph.node("RandomNoise", noise_seed=seed)
    sampler = graph.node("KSamplerSelect", sampler_name="res_multistep" if reference_mode else "euler")
    sigmas = graph.node("BasicScheduler", model=model[0], scheduler="simple",
                        steps=20 if reference_mode else 8, denoise=1.0)
    samples = graph.node("SamplerCustomAdvanced", noise=noise[0], guider=guider[0],
                         sampler=sampler[0], sigmas=sigmas[0], latent_image=av[1])
    images = graph.node("VAEDecode", samples=samples[0], vae=video_vae[0])
    audio = graph.node("VAEDecodeAudio", samples=samples[0], vae=audio_vae[0])
    video = graph.node("CreateVideo", images=images[0], fps=float(FPS), audio=audio[0], bit_depth=8)
    graph.node("SaveVideo", video=video[0], filename_prefix=filename_prefix, format="mp4", codec="h264")
    return graph.to_dict()


def build_t2v(prompt: str, *, width: int = 864, height: int = 480, frames: int = 124,
              seed: int = 0, filename_prefix: str = "easel/videos/minimax_h3") -> dict:
    return _build_video(prompt, width=width, height=height, frames=frames,
                        seed=seed, filename_prefix=filename_prefix)


def build_i2v(prompt: str, *, image_filename: str, width: int = 864, height: int = 480,
              frames: int = 124, seed: int = 0,
              filename_prefix: str = "easel/videos/minimax_h3") -> dict:
    return _build_video(prompt, image_filename=_image_name(image_filename), width=width,
                        height=height, frames=frames, seed=seed, filename_prefix=filename_prefix)


def build_r2v(prompt: str, *, reference_filenames: list[str] | tuple[str, ...],
              ref_image_size: str = "match", width: int = 864, height: int = 480,
              frames: int = 124, seed: int = 0,
              filename_prefix: str = "easel/videos/minimax_h3") -> dict:
    return _build_video(prompt, reference_mode=True,
                        reference_filenames=_references(reference_filenames, required=True),
                        ref_image_size=ref_image_size, width=width, height=height,
                        frames=frames, seed=seed, filename_prefix=filename_prefix)


def build_temporal_guided_video(
    prompt: str, *, guides: list[TemporalGuide] | tuple[TemporalGuide, ...],
    reference_filenames: list[str] | tuple[str, ...] = (), ref_image_size: str = "match",
    width: int = 864, height: int = 480, frames: int = 124,
    seed: int = 0, filename_prefix: str = "easel/videos/minimax_h3",
) -> dict:
    """REF20 temporal guides, graph-tested only; matching still frames are caller-owned."""
    _integer("frames", frames, MIN_FRAMES, MAX_FRAMES)
    return _build_video(prompt, reference_mode=True, guides=_guides(guides, frames),
                        reference_filenames=_references(reference_filenames, required=False),
                        ref_image_size=ref_image_size, width=width, height=height,
                        frames=frames, seed=seed, filename_prefix=filename_prefix)
