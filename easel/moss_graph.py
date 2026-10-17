"""MOSS-Image native HF graph; upstream provenance is retained in NOTICE."""
from __future__ import annotations

import secrets

from .workflow_graph import WorkflowGraph

MODEL_PATH = "/app/ComfyUI/models/moss-image/MOSS-Image-Base"


def text_to_image(*, prompt, width=1024, height=1024, steps=50, batch_size=1,
                  seed=None, model_path=MODEL_PATH, filename_prefix="easel_moss") -> dict:
    """Use upstream sampling with FP8 DiT / CPU text encoding for the image GPU."""
    if not model_path or not model_path.strip():
        raise ValueError("model_path must name a local MOSS-Image HF directory")
    graph = WorkflowGraph()
    loaded = graph.node("MOSSImageHFLoader", model_path=model_path,
                        weight_dtype="fp8_e4m3fn", text_device="cpu")
    if isinstance(prompt, list):
        prompts = [value for value in prompt if value and value.strip()]
        prompt = prompts[0] if len(prompts) == 1 else ""
        if len(prompts) > 1:
            prompt = graph.node("SimplePromptBatcher", prepend="", append="",
                                prompts="\n".join(prompts) + "\n")[0]
    positive = graph.node("CLIPTextEncode", text=prompt, clip=loaded[1])
    negative = graph.node("CLIPTextEncode", text="", clip=loaded[1])
    latent = graph.node("EmptyFlux2LatentImage", width=width, height=height,
                        batch_size=batch_size)
    shifted = graph.node("ModelSamplingSD3", model=loaded[0], shift=5.0)
    guided = graph.node("SkipLayerGuidanceDiTSimple", model=shifted[0],
                        double_layers="", single_layers="22",
                        start_percent=0.0, end_percent=1.0)
    sampled = graph.node(
        "KSampler", model=guided[0], positive=positive[0], negative=negative[0],
        latent_image=latent[0], seed=secrets.randbits(64) if seed is None else seed,
        steps=steps, cfg=3.0, sampler_name="euler", scheduler="sgm_uniform", denoise=1.0,
    )
    decoded = graph.node("VAEDecode", samples=sampled[0], vae=loaded[2])
    graph.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return graph.to_dict()
