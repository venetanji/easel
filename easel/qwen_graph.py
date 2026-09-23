"""Qwen Image 2.1 ComfyUI graph builders."""
from __future__ import annotations

import time

from .workflow_graph import WorkflowGraph

MAX_REFERENCES = 16


def _seed(seed):
    return int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 32)


def _prompt_input(g: WorkflowGraph, prompt):
    prompts = [prompt] if isinstance(prompt, str) else [p for p in prompt if p and p.strip()]
    prompts = prompts or [""]
    if len(prompts) > 1:
        batcher = g.node("SimplePromptBatcher", prepend="", prompts="\n".join(prompts) + "\n",
                         append="")
        return batcher[0]
    return prompts[0]


def _load_models(g: WorkflowGraph, unet_name: str, clip_name: str, vae_name: str):
    unet = g.node("UNETLoader", unet_name=unet_name, weight_dtype="default")
    clip = g.node("CLIPLoader", clip_name=clip_name, type="qwen_image", device="default")
    vae = g.node("VAELoader", vae_name=vae_name)
    return unet, clip, vae


def _sampler_tail(g: WorkflowGraph, *, model, positive, negative, latent, vae,
                  steps, seed, filename_prefix):
    sampled = g.node(
        "KSampler", model=model, positive=positive, negative=negative, latent_image=latent,
        seed=_seed(seed), steps=steps, cfg=1, sampler_name="euler", scheduler="simple",
        denoise=1,
    )
    decoded = g.node("VAEDecode", samples=sampled[0], vae=vae)
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)


def text_to_image(*, unet_name, clip_name, vae_name, prompt, width, height, steps,
                  batch_size=1, seed=None,
                  filename_prefix="easel_t2i") -> dict:
    g = WorkflowGraph()
    unet, clip, vae = _load_models(g, unet_name, clip_name, vae_name)
    encoded = g.node(
        "TextEncodeQwenImage21", clip=clip[0], prompt=_prompt_input(g, prompt),
        negative_prompt="", resolution=1024,
    )
    latent = g.node("EmptyLatentImage", width=width, height=height, batch_size=batch_size)
    _sampler_tail(g, model=unet[0], positive=encoded[0], negative=encoded[1],
                  latent=latent[0], vae=vae[0], steps=steps, seed=seed,
                  filename_prefix=filename_prefix)
    return g.to_dict()


def reference_edit(*, unet_name, clip_name, vae_name, image_filenames, prompt, width=None,
                   height=None, steps, batch_size=1,
                   seed=None, filename_prefix="easel_edit") -> dict:
    if not image_filenames:
        raise ValueError("image_filenames must contain at least one reference")
    if len(image_filenames) > MAX_REFERENCES:
        raise ValueError(f"Qwen Image 2.1 accepts at most {MAX_REFERENCES} reference images")

    g = WorkflowGraph()
    unet, clip, vae = _load_models(g, unet_name, clip_name, vae_name)
    encoded_inputs = {
        f"images.image_{index}": g.node("LoadImage", image=filename)[0]
        for index, filename in enumerate(image_filenames, start=1)
    }
    resolution = max(width, height) if width is not None and height is not None else 1024
    resolution = max(32, int(resolution / 32 + 0.5) * 32)
    encoded = g.node(
        "TextEncodeQwenImage21", clip=clip[0], prompt=_prompt_input(g, prompt),
        negative_prompt="", vae=vae[0], resolution=resolution, **encoded_inputs,
    )
    cached_model = g.node("QwenImage21Cache", model=unet[0], device="auto", dtype="default")
    latent = encoded[2]
    if batch_size > 1:
        latent = g.node("RepeatLatentBatch", samples=latent, amount=batch_size)[0]
    _sampler_tail(g, model=cached_model[0], positive=encoded[0], negative=encoded[1],
                  latent=latent, vae=vae[0], steps=steps, seed=seed,
                  filename_prefix=filename_prefix)
    return g.to_dict()
