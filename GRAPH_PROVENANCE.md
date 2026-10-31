# Graph reuse and compatibility boundary

Easel keeps HTTP/authentication, queue admission, receipts and request validation
separate from its Python ComfyUI graph builders. JavaScript/CLI consumers use the
same HTTP contract. They do not need a port of the graph builder.

## Native YuE2 API (24 October 2026)

`easel/yue2_graph.py` vendors only the full-ABC branch of the creative agent's
`comfy-graph/src/creative_comfy_graph/yue2.py` from the isolated
`creative-skills-yue2-20261022` working tree. This module was not in baseline
commit `c3788aa6e1f954bc13fd4646531991b1096c5786` or a published package.
The source SHA-256 is
`5c5cfd83a4a6c9fe7c30d51b89c90f2d3c0474f03c2d0550269bb2524ef326fe`.
Vendoring matches Easel's existing graph-helper pattern and makes its frozen
Docker build independent of an unpublished external checkout.

`tests/fixtures/yue2/full_abc_graph.json` was generated from that original
builder with the test brief, seed 0, 120-second ceiling and
`filename_prefix='easel/audio/yue2'`. Its SHA-256 is
`bec0de66bf52792cb0347790796b5955c8fba5e7dc73415da77400fd697f1049`.
The test compares every setting and link after adjusting the original zero-based
node IDs to Easel's one-based IDs. No alternate planning modes are exposed.
`SaveAudioAdvanced.format` stays the native **string** wire selector `flac`,
not the execution-time DynamicCombo object. The generated-seconds output sizes
the latent; the request's duration ceiling does not.

`tests/fixtures/yue2/native_schema.json` is copied unchanged from the creative
agent's 22 October capture, SHA-256
`e66b7a31762ce516aee40f09a01d44b20d0973de7200856f48f0ec6525739e82`.
It describes ComfyUI `b5cc883` native interfaces; tests inject the installed
checkpoint choice in memory. Schema descriptions retain upstream ComfyUI terms.
`tone.flac` is a synthetic 48 kHz stereo, 40 ms sine tone produced with ffmpeg,
not YuE2 output. No checkpoint is bundled or downloaded.

The creative agent's separate live CLI evidence includes full-ABC runs up to a
300-second ceiling. That motivates the API ceiling, not a guarantee of exact
duration, natural completion, lyric adherence, or HTTP runtime success.
API tests exercise submission/recovery/cached-output behavior against a fake
ComfyUI service; native readiness was also checked without submitting GPU work.
An additional retrieval-only HTTP/ASGI smoke used the existing successful
30-second CLI pilot (`c0945835-cb2f-4cf1-a081-427da12f3079`) and a temporary
receipt database: real ComfyUI history/fetch, Easel bearer auth, cached FLAC,
byte ranges, ETag 304 and a full ffmpeg decode passed (30 s, 48 kHz stereo,
2,450,966 bytes). This did not submit a prompt, modify production configuration,
verify a new HTTP generation, or test the public reverse-proxy path.
The intended checkpoint's historical upstream revision is
`Comfy-Org/YuE2@2f76ca75e6ee094169de899cc7fc99d6887e2196`, SHA-256
`96fe199377309001ed8cd26a944baeee8cc31a20ba7c36d1d3c0a7e1f4149db6`.
The API checks its installed filename, not a runtime weight hash; it must not
claim byte-identical installed weights or a verified native model revision.

MIT notices for Creative Skills and official templates are retained in `NOTICE`
and `licenses/`; the model weights' CC-BY-NC-4.0 terms and upstream
individual-creator/academic addendum are separate from gateway/recipe code.

## MOSS-Image Base HF plugin (31 October 2026)

`easel/moss_graph.py` adapts the official plugin API workflow at
`OpenMOSS/MOSS-Image@c1355e6b1986dfe8fba3c0455783a05899212342`,
`comfyui/moss-image/example_workflows/moss-image-api.json`.
The upstream workflow and the installed plugin's workflow have SHA-256
`0b293d18872836bde5dfa9d71acd5bec613f5a805bb44e066f928ee6bb2c5995`.
`tests/fixtures/moss/plugin-api.json` preserves its parsed JSON with ASCII
escaping and formatting changes only. The graph test compares all settings and
links independently of node numbering, changing the HF path, precision,
text-encoder device, prompt and output prefix to match Easel.

The source repository's Apache-2.0 license is preserved verbatim in
`licenses/MOSS-Image-Apache-2.0.txt`, SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
No plugin implementation code is bundled. Easel adds its own graph helper,
request-controlled dimensions/batching/seed/steps, prompt fan-out and durable
image receipts. Reference editing and variations are rejected before uploads.

This is the existing **Base HF-directory plugin**, not native MOSS SFT.
In particular, its `ModelSamplingSD3` timestep scaling must not be carried over
unchanged to a future native implementation. The model directory's download
metadata records `OpenMOSS-Team/MOSS-Image-Base` revision
`1ceede80badddd6a7812069619edaa5633587192`; this is not a runtime weight hash.
A 512x512, 20-step, seed-42 direct ComfyUI smoke completed in 34.851 seconds
on the 12 GiB image GPU using CPU text encoding. The PNG was visually inspected.
This establishes basic generation, not editing, native layout support, aligned
sprites, transparency or quality guarantees.
Weight shard headers/lengths and index completeness were checked; no full-file
weight hashes were calculated. ComfyUI revision was
`b5cc8830279eae909a59de030af1e50761c36751`.

After deploying Easel, the public authenticated API returned a durable HTTP 202
receipt and served a valid 512x512 PNG for the same seed-42 graph. ComfyUI reused
cached sampling for this first API round trip. A separate request omitted size
and steps and used seed 43 to force fresh sampling: the runtime graph contained
1024x1024 and 50 steps, completed in 57.76 seconds, and the authenticated job
content route served a 519346-byte PNG. Its SHA-256 is
`b7e2b64bddf46e63fc2629b3fddcbbfcf0e0f21a0ae48256d958792c357f347c`.
Both POSTs were issued once and saved before polling. Unauthenticated public
model discovery returned HTTP 401. One default-size success is not a memory or
throughput guarantee for every request.

## MOSS SFT DMD Adapter (31 October 2026)

The separate `moss-image-sft-8step` registry entry uses the same HF plugin with
the actual SFT transformer, BF16 loading, and `LoraLoaderModelOnly` at strength
1.0. The eight-step, no-CFG recommendation comes from the pinned
[lodestones model card](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/blob/f95df83b5e90ae91ea412434b3f9b7f1c8c6314f/README.md).
Its reference
[demo](https://huggingface.co/spaces/hugging-apps/moss-image-8-step-dmd/blob/2ff429beafaee462b97f6c34238959b19d270f05/app.py)
confirms standard flow Euler sampling at shift 5 and says the old data-space
loop is unused. That demo selects `lora_latest` at 0.7; this profile uses the
user-requested **lora.safetensors at 1.0** per the model card. No demo or custom
sampler implementation is bundled. The existing plugin uses SD3 timestep
scaling with Euler / sgm_uniform, which does not promise exact equivalence to
the demo's scheduler or seeds.

The official SFT revision is `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`.
All three transformer shard downloads and the selected adapter were checked
against their Hub SHA-256 values. Base text encoder/tokenizer/VAE repository
blobs match SFT and are reused through relative symlinks. The adapter SHA-256 is
`4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe`.
All 252 rank-128 adapter targets matched the SFT tensor shapes; the runtime log
confirmed **252 patches attached**. An eight-step 1024x1024, seed-43 generation
completed in 30.403 seconds from cold loading. Independent same-prompt/seed
SFT controls used 50 steps/CFG 3/SLG and eight steps/CFG 1/no SLG/no LoRA.
Saved graphs, receipts, histories and images support these observations.
The adapter is an upstream work in progress; a crisp sample is not universal
quality equivalence, a pixel-grid guarantee or editing support.

The deployed public API then accepted a seed-44 request with omitted size/steps,
returned HTTP 202, completed the durable receipt and served a valid 867508-byte
1024x1024 PNG. Captured runtime inputs confirm eight steps, CFG 1, BF16 and no
SLG; the sampler ran fresh while model/text nodes were cached. ComfyUI execution
took 6.679 seconds; acceptance-to-polling/download was about nine seconds.
The PNG SHA-256 is
`9e0bdad023e03f95b5fceeb2a3bc1431a778990e62080f338f90b0ffd1cccfa0`.
Each comparison/API POST was issued once and saved before polling.

## Licensing and maintainer confirmation

On 5 October 2026 the maintainer confirmed that Comfy Graph recipes derive from
official ComfyUI documentation and GitHub workflows, and that the Python
generator is their own GPL code. Easel-owned generators are now explicitly
GPL-3.0-or-later. This resolves the earlier ownership question; it does not
assert a line-by-line map for unpublished historical recipes or change the
technical compatibility boundaries below.

The creative-skills portions retain their original MIT grant. Official
`Comfy-Org/workflow_templates` at the pinned revision in H3_BASELINE.md is MIT;
ComfyUI source/blueprints retain their upstream GPLv3 terms. Full notices are
preserved in [NOTICE](NOTICE) and `licenses/`, including for copyrightable
copied schema descriptions in the offline fixtures. The own-code "or later"
grant does not amend ComfyUI's grant. Model assets retain separate terms.

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

## Recovered source-audio recipes (12 October 2026)

The active Creative Skills main worktree now contains `comfy-graph/` and its
skill adapters, recovered from archived main `0e208b2` and the uncommitted H3
worktree `feat/minimax-h3-t2va-20261002`. See `comfy-graph/RECOVERY.md` in that
repository for the recovery boundary. Easel remains independent of that package
at runtime and never imports temporary worktree paths.

Easel's optional LTX audio branch follows the recovered
`creative_comfy_graph.ltx_profiles.build_video` source-encoding/frozen-mask
topology, restoring the original encoded audio between its sampling passes.
Its H3 audio branch follows the native `MiniMaxH3AddGuide` audio ports in the
pinned ComfyUI source already cited in H3_BASELINE.md. Both branches mux the
trimmed normalized source instead of model-decoded audio. This updates the
source-availability boundary; it does not claim a new immutable publication,
byte-identical generators, GPU execution, or perceptual lip-sync validation.

## H3 cloud baseline

`easel/h3_graph.py` adapts the unpublished October 2 H3 prototype into Easel's
graph helper, with independent recipe fixtures and a captured native runtime
schema. Bounded HTTP integration is present. Temporal guidance has offline
graph-contract evidence, not a completed continuity pilot. See [H3_BASELINE.md](H3_BASELINE.md)
for pinned ComfyUI/model/workflow pointers and cloud implementation gates.
