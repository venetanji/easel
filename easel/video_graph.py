"""LTX-2.5 text-to-video and image-to-video ComfyUI graphs."""
from __future__ import annotations

import time

from .workflow_graph import WorkflowGraph

VIDEO_SIZES = {
    "512x320": (512, 320),
    "640x384": (640, 384),
    "768x512": (768, 512),
    "1280x720": (1280, 720),
    "720x1280": (720, 1280),
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
}

VIDEO_MODEL_ID = "ltx-2.5"
UNET_NAME = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
VIDEO_VAE_NAME = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_NAME = "ltx-2.5-audio-vae-bf16.safetensors"
TEXT_ENCODER_NAME = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
UPSCALE_MODEL_NAME = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

FPS = 24
NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"
SIGMAS_PASS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"


def parse_video_size(value: str | None) -> tuple[int, int]:
    size = (value or "1280x720").strip().lower()
    try:
        return VIDEO_SIZES[size]
    except KeyError:
        allowed = ", ".join(VIDEO_SIZES)
        raise ValueError(f"size must be one of: {allowed}")


def _ic_guide(graph, conditioning, latent, image, vae, downscale, strength):
    return graph.node("LTXAddVideoICLoRAGuide", positive=conditioning[0],
                  negative=conditioning[1], vae=vae, latent=latent, image=image,
                  frame_idx=0, strength=strength, latent_downscale_factor=downscale,
                  crop="disabled", use_tiled_encode=True, tile_size=256, tile_overlap=64)


def text_to_video(*, prompt: str, seconds: int, width: int, height: int,
                  filename_prefix: str, input_image: str | None = None,
                  seed: int | None = None, loras: list[tuple[str, float]] | None = None,
                  motion_speed: float | None = None,
                  ic_lora: tuple[str, float] | None = None,
                  reference_image: str | None = None, reference_strength: float = 1.0) -> dict:
    """Build the ComfyUI graph used by the native LTX-2.5 T2V and I2V templates."""
    g = WorkflowGraph()
    model = g.node("UNETLoader", unet_name=UNET_NAME, weight_dtype="default")
    for filename, strength in loras or []:
        model = g.node("LoraLoaderModelOnly", model=model[0],
                       lora_name=filename, strength_model=strength)
    ic_model = None
    if ic_lora:
        if not reference_image or seconds < 5:
            raise ValueError("Ingredients needs a reference sheet and at least 5 seconds")
        ic_model = g.node("LTXICLoRALoaderModelOnly", model=model[0],
                          lora_name=ic_lora[0], strength_model=ic_lora[1])
        model = ic_model
    video_vae = g.node("VAELoader", vae_name=VIDEO_VAE_NAME)
    audio_vae = g.node("VAELoader", vae_name=AUDIO_VAE_NAME)
    clip = g.node("CLIPLoader", clip_name=TEXT_ENCODER_NAME, type="ltxv", device="default")

    positive = g.node("CLIPTextEncode", text=prompt, clip=clip[0])
    negative = g.node("CLIPTextEncode", text=NEGATIVE_PROMPT, clip=clip[0])
    conditioning = g.node(
        "LTXVConditioning", positive=positive[0], negative=negative[0],
        frame_rate=FPS / motion_speed if motion_speed is not None else FPS,
    )

    # LTX-2.5's native workflow samples at half resolution, then upsamples 2x.
    length = seconds * FPS + 1
    reference = None
    if ic_model:
        sheet = g.node("LoadImage", image=reference_image)
        scaled = g.node("ImageScale", image=sheet[0], upscale_method="lanczos",
                        width=width, height=height, crop="disabled")
        reference = g.node("RepeatImageBatch", image=scaled[0], amount=length)
    latent = g.node(
        "EmptyLTXVLatentVideo", width=width // 2, height=height // 2,
        length=length, batch_size=1,
    )
    image = None
    if input_image:
        loaded = g.node("LoadImage", image=input_image)
        resized = g.node(
            "ResizeImageMaskNode", **{
                "input": loaded[0],
                "resize_type": "scale longer dimension",
                "resize_type.longer_size": 1536,
                "scale_method": "lanczos",
            },
        )
        image = g.node("LTXVPreprocess", image=resized[0], img_compression=18)
        latent = g.node(
            "LTXVImgToVideoInplace", vae=video_vae[0], image=image[0], latent=latent[0],
            strength=0.7, bypass=False,
        )

    audio = g.node(
        "LTXVEmptyLatentAudio", frames_number=length, frame_rate=FPS,
        batch_size=1, audio_vae=audio_vae[0],
    )
    stage1_conditioning = conditioning
    video_samples = latent[0]
    if reference:
        stage1_conditioning = _ic_guide(g, conditioning, video_samples, reference[0],
            video_vae[0], ic_model[1], reference_strength)
        video_samples = stage1_conditioning[2]
    av_latent = g.node("LTXVConcatAVLatent", video_latent=video_samples, audio_latent=audio[0])

    seed = seed if seed is not None else int(time.time() * 1000) % (2 ** 32)
    guider = g.node(
        "LTXVDualCFGGuider", model=model[0], positive=stage1_conditioning[0],
        negative=stage1_conditioning[1], video_cfg=1, audio_cfg=1,
    )
    sample1 = g.node(
        "SamplerCustomAdvanced",
        noise=g.node("RandomNoise", noise_seed=seed)[0],
        guider=guider[0],
        sampler=g.node("KSamplerSelect", sampler_name="euler_ancestral")[0],
        sigmas=g.node("ManualSigmas", sigmas=SIGMAS_PASS1)[0],
        latent_image=av_latent[0],
    )
    split1 = g.node("LTXVSeparateAVLatent", av_latent=sample1[0])
    video_samples = split1[0]
    if reference:
        cropped = g.node("LTXVCropGuides", positive=stage1_conditioning[0],
                         negative=stage1_conditioning[1], latent=video_samples)
        video_samples = cropped[2]

    upscaler = g.node("LatentUpscaleModelLoader", model_name=UPSCALE_MODEL_NAME)
    video_latent = g.node(
        "LTXVLatentUpsampler", samples=video_samples,
        upscale_model=upscaler[0], vae=video_vae[0],
    )
    if image:
        video_latent = g.node(
            "LTXVImgToVideoInplace", vae=video_vae[0], image=image[0], latent=video_latent[0],
            strength=1.0, bypass=False,
        )
    stage2_conditioning = conditioning
    video_samples = video_latent[0]
    if reference:
        stage2_conditioning = _ic_guide(g, conditioning, video_samples, reference[0],
            video_vae[0], ic_model[1], reference_strength)
        video_samples = stage2_conditioning[2]
    av_latent = g.node("LTXVConcatAVLatent", video_latent=video_samples, audio_latent=split1[1])

    guider2 = g.node(
        "LTXVDualCFGGuider", model=model[0], positive=stage2_conditioning[0],
        negative=stage2_conditioning[1], video_cfg=1, audio_cfg=1,
    )
    sample2 = g.node(
        "SamplerCustomAdvanced",
        noise=g.node("RandomNoise", noise_seed=seed + 1)[0],
        guider=guider2[0],
        sampler=g.node("KSamplerSelect", sampler_name="euler_ancestral")[0],
        sigmas=g.node("ManualSigmas", sigmas=SIGMAS_PASS2)[0],
        latent_image=av_latent[0],
    )
    split2 = g.node("LTXVSeparateAVLatent", av_latent=sample2[0])
    video_samples = split2[0]
    if reference:
        cropped = g.node("LTXVCropGuides", positive=stage2_conditioning[0],
                         negative=stage2_conditioning[1], latent=video_samples)
        video_samples = cropped[2]

    frames = g.node(
        "VAEDecodeTiled", samples=video_samples, vae=video_vae[0],
        tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16,
    )
    generated_audio = g.node(
        "LTXVAudioVAEDecode", samples=split2[1], audio_vae=audio_vae[0],
    )
    video = g.node("CreateVideo", images=frames[0], audio=generated_audio[0], fps=FPS)
    g.node(
        "SaveVideo", video=video[0], filename_prefix=filename_prefix,
        format="auto", codec="auto",
    )
    return g.to_dict()
