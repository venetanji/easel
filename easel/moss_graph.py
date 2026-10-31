# MOSS-Image plugin workflow adaptation; see NOTICE and GRAPH_PROVENANCE.md.
"""MOSS-Image graphs for the installed Hugging Face directory plugin."""
from __future__ import annotations

import time

from .workflow_graph import WorkflowGraph


def text_to_image(*, model_path, prompt, width, height, steps, batch_size=1,
                  seed=None, filename_prefix="easel_moss_t2i", distilled_lora=None) -> dict:
    g = WorkflowGraph()
    loaded = g.node("MOSSImageHFLoader", model_path=model_path,
                    weight_dtype="bf16" if distilled_lora else "fp8_e4m3fn", text_device="cpu")
    model = loaded[0]
    if distilled_lora:
        model = g.node("LoraLoaderModelOnly", model=model, lora_name=distilled_lora,
                       strength_model=1.0)[0]
    prompts = [prompt] if isinstance(prompt, str) else [p for p in prompt if p and p.strip()]
    prompts = prompts or [""]
    text = prompts[0]
    if len(prompts) > 1:
        text = g.node("SimplePromptBatcher", prepend="", prompts="\n".join(prompts) + "\n",
                      append="")[0]
    positive = g.node("CLIPTextEncode", clip=loaded[1], text=text)
    negative = g.node("CLIPTextEncode", clip=loaded[1], text="")
    latent = g.node("EmptyFlux2LatentImage", width=width, height=height,
                    batch_size=batch_size)
    # This plugin uses SD3 timestep scaling; the newer native MOSS path differs.
    model = g.node("ModelSamplingSD3", model=model, shift=5)[0]
    if not distilled_lora:
        model = g.node("SkipLayerGuidanceDiTSimple", model=model, double_layers="",
                       single_layers="22", start_percent=0, end_percent=1)[0]
    sampled = g.node(
        "KSampler", model=model, positive=positive[0], negative=negative[0],
        latent_image=latent[0], seed=int(seed) if seed is not None else
        int(time.time() * 1000) % (2 ** 32), steps=steps, cfg=1 if distilled_lora else 3,
        sampler_name="euler", scheduler="sgm_uniform", denoise=1,
    )
    decoded = g.node("VAEDecode", samples=sampled[0], vae=loaded[2])
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return g.to_dict()
