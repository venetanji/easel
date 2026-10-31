"""Native full-ABC YuE2 recipe adapted from Creative Comfy Graph; see NOTICE."""
from __future__ import annotations

from .workflow_graph import WorkflowGraph

MODEL_ID = "yue2"
RECIPE_VERSION = "yue2.full-abc.v1"
CHECKPOINT = "yue2_3b_int8_convrot.safetensors"
DEFAULT_DURATION = 120.0
MAX_DURATION = 300.0
MAX_SEED = 2**53 - 1


def build_text_to_music(style: str, lyrics: str, *, max_duration: float = DEFAULT_DURATION,
                        seed: int = 0, filename_prefix: str = "easel/audio/yue2") -> dict:
    graph = WorkflowGraph()
    checkpoint = graph.node("CheckpointLoaderSimple", ckpt_name=CHECKPOINT)
    score = graph.node("YuE2GenerateABC", clip=checkpoint[1], style=style,
                       lyrics=lyrics, seed=seed, mode="full", max_abc_tokens=8192,
                       temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005,
                       penalty_window=100)
    music = graph.node("YuE2GenerateMusic", clip=checkpoint[1], style=style,
                       lyrics=lyrics, abc=score[0], seed=seed, mode="full",
                       max_duration=max_duration, temperature=1.0, top_p=0.95,
                       top_k=100, repetition_penalty=1.2, cfg_scale=1.0)
    latent = graph.node("EmptyYuE2LatentAudio", seconds=music[1], batch_size=1)
    negative = graph.node("ConditioningZeroOut", conditioning=music[0])
    samples = graph.node("KSampler", model=checkpoint[0], seed=seed, steps=32,
                         cfg=1.0, sampler_name="dpm_2", scheduler="sgm_uniform",
                         positive=music[0], negative=negative[0], latent_image=latent[0],
                         denoise=1.0)
    audio = graph.node("VAEDecodeAudio", samples=samples[0], vae=checkpoint[2])
    graph.node("SaveAudioAdvanced", audio=audio[0], filename_prefix=filename_prefix, format="flac")
    return graph.to_dict()
