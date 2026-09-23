"""Flux.2 Klein ComfyUI graph builders (vendored).

Self-contained so nothing imports the creative-skills tree at runtime (that tree
is pulled/overwritten from the overlord repo). Mirrors the known-good graphs in
creative-skills/comfyui/scripts/flux2.py:

  - Flux.2 uses CLIP `qwen_3_8b_fp8mixed.safetensors` with type="flux2" (NOT gemma).
  - CFG is 1.0; the negative branch is inert at that cfg but kept for graph fidelity.
  - Reference editing: VAE-encode each ref, chain one ReferenceLatent per ref onto
    BOTH positive and negative conditioning; negative = ConditioningZeroOut(positive).
  - LatentAddNoise is NOT installed here: pass the latent straight to the sampler.
"""
from __future__ import annotations

import time

from .workflow_graph import NodeRef, WorkflowGraph

CLIP_NAME = "qwen_3_8b_fp8mixed.safetensors"
VAE_NAME = "flux2-vae.safetensors"
DEFAULT_STEPS = 8


def _seed(seed):
    return int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 32)

def _norm_prompts(prompt) -> list[str]:
    """Accept a single prompt or a list; drop blank entries. Never empty."""
    if isinstance(prompt, str):
        return [prompt]
    parts = [p for p in prompt if p and p.strip()]
    return parts or [""]


def _positive(g, clip, prompt):
    """Build the positive conditioning. >1 prompt -> SimplePromptBatcher fan-out
    (one output per line); else a plain CLIPTextEncode."""
    prompts = _norm_prompts(prompt)
    if len(prompts) > 1:
        batcher = g.node("SimplePromptBatcher", prepend="",
                         prompts="\n".join(prompts) + "\n", append="")
        return g.node("CLIPTextEncode", text=batcher[0], clip=clip), True
    return g.node("CLIPTextEncode", text=prompts[0], clip=clip), False


def _load_models(g: WorkflowGraph, unet_name: str, clip_name: str):
    unet = g.node("UNETLoader", unet_name=unet_name, weight_dtype="default")
    vae = g.node("VAELoader", vae_name=VAE_NAME)
    clip = g.node("CLIPLoader", clip_name=clip_name, type="flux2", device="default")
    return unet, vae, clip


def _sampler_tail(g, *, model, positive, negative, vae, width, height,
                  steps, batch_size, seed, filename_prefix):
    latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch_size)
    sampler = g.node("KSamplerSelect", sampler_name="euler")
    sched = g.node("Flux2Scheduler", steps=steps, width=width, height=height)
    noise = g.node("RandomNoise", noise_seed=_seed(seed))
    guider = g.node("CFGGuider", model=model, positive=positive, negative=negative, cfg=1.0)
    sampled = g.node("SamplerCustomAdvanced", noise=noise[0], guider=guider[0],
                     sampler=sampler[0], sigmas=sched[0], latent_image=latent[0])
    decoded = g.node("VAEDecode", samples=sampled[0], vae=vae)
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)


def text_to_image(*, unet_name, prompt, width, height, clip_name=CLIP_NAME,
                  steps=DEFAULT_STEPS, batch_size=1, seed=None,
                  filename_prefix="easel_t2i") -> dict:
    g = WorkflowGraph()
    unet, vae, clip = _load_models(g, unet_name, clip_name)
    pos, multi = _positive(g, clip[0], prompt)
    # For a fan-out batch the negative must be per-item; ConditioningZeroOut mirrors
    # the list. Single prompt keeps the empty-string negative.
    neg = g.node("ConditioningZeroOut", conditioning=pos[0]) if multi \
        else g.node("CLIPTextEncode", text="", clip=clip[0])
    _sampler_tail(g, model=unet[0], positive=pos[0], negative=neg[0], vae=vae[0],
                  width=width, height=height, steps=steps, batch_size=batch_size,
                  seed=seed, filename_prefix=filename_prefix)
    return g.to_dict()


def reference_edit(*, unet_name, image_filenames, prompt, width=None, height=None,
                   clip_name=CLIP_NAME, steps=DEFAULT_STEPS, batch_size=1, seed=None,
                   filename_prefix="easel_edit") -> dict:
    """N-reference Flux.2 edit. If width/height are omitted, the output size is
    derived from the first reference (scaled to 1MP) via GetImageSize."""
    if not image_filenames:
        raise ValueError("image_filenames must contain at least one reference")
    g = WorkflowGraph()
    unet, vae, clip = _load_models(g, unet_name, clip_name)
    pos, _multi = _positive(g, clip[0], prompt)
    neg = g.node("ConditioningZeroOut", conditioning=pos[0])

    encoded = []
    first_scaled = None
    for fn in image_filenames:
        ref = g.node("LoadImage", image=fn)
        scaled = g.node("ImageScaleToTotalPixels", image=ref[0],
                        upscale_method="nearest-exact", megapixels=1, resolution_steps=1)
        if first_scaled is None:
            first_scaled = scaled
        encoded.append(g.node("VAEEncode", pixels=scaled[0], vae=vae[0]))

    pos_ref = pos
    for enc in encoded:
        pos_ref = g.node("ReferenceLatent", conditioning=pos_ref[0], latent=enc[0])
    neg_ref = neg
    for enc in encoded:
        neg_ref = g.node("ReferenceLatent", conditioning=neg_ref[0], latent=enc[0])

    if width is None or height is None:
        size = g.node("GetImageSize", image=first_scaled[0])
        out_w, out_h = size[0], size[1]
    else:
        out_w, out_h = width, height

    _sampler_tail(g, model=unet[0], positive=pos_ref[0], negative=neg_ref[0], vae=vae[0],
                  width=out_w, height=out_h, steps=steps, batch_size=batch_size,
                  seed=seed, filename_prefix=filename_prefix)
    return g.to_dict()
