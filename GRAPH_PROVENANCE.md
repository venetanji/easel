# Graph reuse and compatibility boundary

Easel keeps HTTP/authentication, queue admission, receipts and request validation
separate from its Python ComfyUI graph builders. JavaScript/CLI consumers use the
same HTTP contract. They do not need a port of the graph builder.

## What is inherited

`flux_graph.py` explicitly documents vendoring from creative-skills' Flux2
builder. Its small `WorkflowGraph` helper was later extracted to
`easel/workflow_graph.py`; node numbering/link handling is adapted rather than
byte-identical to creative-skills' `core.py`.

The video builder shares a coarse → latent upscale → refine pattern and sigma
schedules with published Comfy Graph, but Easel uses LTX-2.5-specific UNET,
Gemma4 encoder, VAEs and upscaler, DualCFG guider and its own request/adapter
validation. Exact source lineage to the unpublished canonical LTX-2.5 runtime
cannot be established from the available repository. This remains an interim
server-owned implementation, as already stated in `VIDEO_LORAS.md`.

The inspected published creative-skills tree is
[`67f185c8`](https://github.com/venetanji/creative-skills/tree/67f185c8f95182690e1ac56e734aa649f71b4df6).
Its `comfyui/scripts/ltx2.py` is LTX-2.3. The newer LTX-2.5 profile/runtime described
in [creative-skills PR #11](https://github.com/venetanji/creative-skills/pull/11)
was explicitly excluded from that publication. Do not substitute the old
checkpoint/sampler/transition behavior or claim the canonical runtime was
imported or migrated.

## Timed still-image guides

The current API uses `LTXVAddGuide`, verified against ComfyUI
[`e9027f2b`](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/comfy_extras/nodes_lt.py#L251-L519)
and its [LTX-2.5 first/last-frame blueprint](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/blueprints/First%20%26%20Last%20Frame%20to%20Video%20(LTX-2.5).json).
That blueprint uses Easel's same int8-convrot UNET and generated audio. It is
single-stage; the new two-stage composition is graph-contract-tested pending
an opted-in GPU pilot.

Single-image indices are exact pixel-frame positions. Multi-frame guidance has
different snapping/causal semantics, so animated images are rejected. The node
resizes/center-crops a still image to its current latent dimensions. All guide
chains are added before AV concatenation; all appended latent slots and
conditioning metadata are cropped before upscale and before decode. Indices
are validated against the original requested length, not an intermediate latent
that already contains appended guide slots. Per-guide strengths and seed zero
are preserved in both passes.

## IC profiles stay separate

IC-LoRAs use their own loader metadata and guide-node contract. Full control-video
reinsertion in refinement is not a generic rule: published Union guidance and
legacy IC history specifically distinguish it from ordinary image anchors.
Current Easel Ingredients retains its documented, separately execution-tested
reference-sheet behavior. No additional installed-but-unsupported IC adapter is
enabled by timed still-image guides.

A future canonical extraction should first publish immutable LTX-2.5 source and
graph fixtures, then migrate the registry, model profiles and profile-specific
pass policy behind these tested API boundaries. Discovery must continue to
separate code support, installation/node availability, execution evidence and
visual quality. Credentials, media uploads and job persistence stay with Easel.
