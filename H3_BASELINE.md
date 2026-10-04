# H3 and temporal-guide cloud baseline (PR #3)

This branch supplies offline graphs, captured runtime contracts, and bounded
HTTP integration for `minimax-h3` on the video backend. Qwen and LTX-2.5
behavior and defaults are unchanged. Native runtime discovery gates each
selected H3 graph before queue admission or uploads. The local owner reviews,
pilots, and deploys these changes; GPU execution and visual review remain
rollout gates.

## Available code and fixtures

`easel/h3_graph.py` ports the previously explored, unpublished Comfy Graph
recipes into Easel's own `WorkflowGraph`, without temporary paths or a new
package dependency:

- `build_t2v`: native FL8 text to generated audio/video.
- `build_i2v`: FL8 with one first-frame anchor.
- `build_r2v`: base REF20 with 1-2 ordered semantic Picture references.
- `build_temporal_guided_video`: REF20 with chained `MiniMaxH3AddGuide`
  conditioning. `TemporalGuide` holds ordered managed image names and a
  target pixel-frame position. Baseline limits: 1-3 nonoverlapping groups,
  each containing exactly 1, 5, 22, or 39 images.

The first three recipes match independent prototypes in `tests/fixtures/h3/`.
`runtime_nodes.json` captures 19 registered node schemas. Loader choices are
filtered to H3; LoadImage basenames are synthetic, never private uploads.
These fixtures are offline contracts, not runtime availability probes.

```python
from easel.h3_graph import TemporalGuide, build_temporal_guided_video

graph = build_temporal_guided_video(
    "Continue the same motion and camera direction, one uninterrupted shot.",
    guides=[TemporalGuide(tuple(f"tail_{index:02}.png" for index in range(22)), 0)],
    frames=124,
    seed=0,
)
```

This builds JSON only. The names must later be mapped from validated uploads
to managed ComfyUI input names, not accepted as paths from an HTTP client.

## Current native ComfyUI pointers

Inspected October 4, 2026: runtime revision
`b5cc8830279eae909a59de030af1e50761c36751`; unmodified native H3 node SHA256
`108ceff11e5a5c9a34a57bbc380447cf9941beb673a1bd421abc818de54ccd5c`.
Use these pinned primary sources, not paid/cloud partner nodes:

- [Native H3 nodes](https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/comfy_extras/nodes_minimax_h3.py):
  `align_frame_count`, `temporal_shape`, `MiniMaxH3ImageToVideo`,
  `MiniMaxH3ReferenceToVideo`, `MiniMaxH3AddGuide`, `MiniMaxH3SigmaShift`.
- [Native model](https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/comfy/ldm/minimax/model.py):
  joint AV latents, timeline scaling, and keyframe conditioning.
- [V3 dynamic input expansion](https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/comfy_api/latest/_io.py):
  dotted `ref_images.ref_image_0` / `ref_images.ref_image_1` API ports.
- [Native I2V blueprint](https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/blueprints/Image%20to%20Video%20(MiniMax%20H3).json).
- [Multiframe workflow](https://github.com/Comfy-Org/workflow_templates/blob/0e5c5efb32ba6f3365d6da07da64aaf668157042/templates/video_minimax_h3_multiframe_reference.json).
- [R2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/0e5c5efb32ba6f3365d6da07da64aaf668157042/templates/video_minimax_h3_r2v.json).

Inspect links and active switch branches, not filenames/widget defaults.
UI workflows need API-graph conversion and cannot be submitted directly.
The installed template packages were 0.11.68 / JSON 0.1.94. The installed
multiframe JSON SHA256 was
`f6a30c36fbb1adedc9a0e141ff95b9cf043d08c15350591fab2ddf3a6a90a126`;
the separately pinned upstream file is
`a0526a9f0d67f3fac3ad48bda26c86aa6871c0322a5034534fdfa39d9637e641`.
They are not claimed to be byte-identical. The captured installed workflow
uses three chained AddGuide nodes with the reference-family recipe.

The runtime also exposes `EmptyMiniMaxH3LatentAV`,
`MiniMaxH3FunControlNetApply`, and reference-video/audio ports. Their presence
does not enable those features here. Never fall back to `ComfyCloudMiniMaxH3*`.

## Model-specific boundaries

- FL8: FL-family UNET plus fixed 8-step turbo LoRA, SigmaShift video 12/audio 3,
  Euler/simple 8 steps. REF20: REF-family UNET, res_multistep/simple 20 steps,
  no FL adapter or SigmaShift. Exact curated asset names are constants in
  `h3_graph.py`; no client-supplied model, sampler, or LoRA paths.
- Shared: Qwen3VL-32B encoder with CLIP type `minimax`, explicit installed
  INT8-convrot video VAE, FP32 audio VAE. Both decoders consume the same joint
  sampled AV latent, producing 24 FPS MP4/H.264 plus generated audio.
- Exact output frames: `17k+5`, 124..362, default 124 (5.166667 seconds).
  Never apply LTX's `seconds*24+1` or silently round H3 duration.
- Seed: 0..`2**64-1`; one sampling pass, no LTX seed+1 restriction. Preserve
  JavaScript seeds as exact decimal strings.
- Native dimension bounds are not VRAM/quality evidence. Retained studies
  used 864x480/124 frames; start with conservative HTTP presets.
- Semantic Picture references are not timeline constraints. Native guide
  batches of 2-4 frames silently become their first image; other off-grid
  batches are truncated. This baseline rejects these cases.
- H3 AddGuide returns CONDITIONING only and retains the original AV latent.
  There are no LTX appended guide tokens, CropGuides, or per-guide strengths.
- ImageBatch may resize mismatched inputs. Reject mismatched dimensions in
  the HTTP layer before uploads rather than relying on that fallback.

A 22-frame tail at new frame zero is a bounded continuity pilot proposal.
Keep frame order and 24 FPS, then inspect motion/identity/camera continuity
and frame-accurate overlap trimming/blending. No seamless-join guarantee.
Source-audio guidance and audio seam handling remain unsupported here.

## Cloud implementation checklist

Work on this PR's branch in focused commits:

- [x] Register `minimax-h3` on the video backend and add model-specific,
  authenticated capabilities without changing existing models/defaults.
  Validate model selection; do not silently ignore a capability query.
- [x] Add H3-only explicit `frames`, default 124. Reject explicitly supplied
  `seconds` until an exact mapping is documented; distinguish omission from
  FastAPI's LTX default. Include actual frame/FPS/duration metadata in H3
  receipts without breaking existing receipt/content APIs.
- [x] Define typed semantic-reference and temporal-group metadata. Suggested
  group shape: `{frame_index, image_indices}` referencing repeated still
  uploads in the same request, not paths, URLs, or arbitrary graph JSON.
  Reject unknown/empty fields, orphan/duplicate indices, overlaps, strengths,
  invalid bounds, conflicts, and off-grid counts before upstream work.
- [x] Validate PNG/JPEG bytes, stillness, safe decode, matching dimensions per
  guide, aggregate byte/pixel limits, frame/node counts, and safe managed
  input names. Use each uploaded image exactly as declared. The baseline
  only accepts PNG/JPEG names; do not silently pass WebP through it.
- [x] Check selected native node schemas/assets; separate code support,
  installed/node-compatible, GPU-executed, and visually reviewed evidence.
  Discovery is read-only and must never load models or submit prompts.
- [x] Route FL8 T2V/I2V, REF20 R2V, and experimental REF20 temporal guidance
  without mixing profiles. Reject LTX camera/Ingredients/motion controls and
  unsupported source audio, control video, negative/CFG, or custom sampling.
- [x] Preserve authentication, one-owner queue admission, durable receipts,
  read-only polling/content retrieval, and no automatic POST retry. Validate
  before upload/queue work; do not replace lost or timed-out receipts.
- [x] Replace the baseline's not-registered test with actual HTTP integration
  tests. Cover FastAPI -> fake-Comfy graphs, capabilities, invalid input with
  zero upstream mutation, cross-model receipts/content, and LTX regressions.
- [x] Keep offline tests runnable without Docker, GPU assets, OpenClaw,
  private files, network, or credentials. Existing cross-client tests remain
  explicit opt-in checks, not silently skipped integration evidence.
- [x] Report changed files, tests, live-test gaps, and final commit. Do not
  merge, generate paid media, restart services, or deploy from the cloud
  session; local review, bounded GPU verification, and deployment follow.

## HTTP contract

`GET /v1/videos/capabilities?model=minimax-h3` requires the same authentication
as generation. Omit `model` for existing LTX capabilities; unknown models fail
with `model_not_found`. Discovery reads only the selected native node schemas
and curated asset chooser values. Per-profile `supported` means this code has
a recipe; `available` means its native runtime contract and assets passed.
`validation: graph_contract_tested`, `gpu_executed: false`, and
`visually_reviewed: false` describe this integration's evidence, independent
of historical native recipe studies. Discovery never submits or loads models.
`GET /v1/videos/loras?model=minimax-h3` returns an empty model-specific
catalog: its curated FL8 adapter is fixed by the recipe, not client-selectable.
An omitted model still returns the existing LTX catalog; unknown models fail.

`POST /v1/videos` accepts multipart `model=minimax-h3`, a nonblank `prompt`,
optional unsigned decimal `seed` (0..18446744073709551615), `frames` (default
124, exactly 17k+5 in 124..362), and `size` (only 864x480, also the default).
Explicit `seconds` is rejected, including LTX's default value when explicitly
sent. H3 never rounds duration or applies LTX's second-pass seed increment.

Choose conditioning through these typed controls:

- Omit images and metadata for FL8 text to video.
- One `input_reference` PNG/JPEG still selects FL8 first-frame image to video.
  It cannot be combined with the following REF20 controls.
- Repeated `images` uploads plus `semantic_references`, a JSON array such as
  `[{"image_index":0},{"image_index":1}]`, select ordered semantic Picture
  references. There are at most two; these are not timeline anchors.
- `temporal_groups` is a JSON array such as
  `[{"frame_index":0,"image_indices":[0,1,2,3,4]}]` over repeated `images`
  uploads. There are at most three nonoverlapping groups, each containing
  exactly 1, 5, 22, or 39 images in declared order. Each group must fit entirely
  within the requested output frames and contain images of identical decoded
  dimensions after EXIF orientation (matching native LoadImage). Temporal
  groups use experimental REF20 and may coexist with
  separately indexed semantic references.

Each `images` index must be used exactly once across semantic references and
all temporal groups. Unknown fields, repeated scalar form fields, repeated
JSON keys, empty arrays, paths/URLs, orphan or duplicate indices, guide
strengths, LTX controls, source audio, control video, negative/CFG controls,
and custom model/sampling controls are rejected before runtime discovery.
Multipart file names are discarded; managed PNG/JPEG basenames are generated
by the server. Returned Comfy names must also be safe, unique basenames.
Uploads must decode as still PNG/JPEG with matching MIME, at most 32 MiB
combined, 32 million pixels each, and 64 million pixels combined. The maximum
119 uploads yields a bounded graph (at most 256 nodes); native dimension
bounds do not widen the conservative output preset.

Receipts retain the existing `video_*` identifier and polling/content routes.
New H3 IDs add optional `:frames` to the encoded-model component; this remains
within the published Media MCP job ID character contract. Legacy IDs retain
their original shape and parse behavior. H3 create/poll/queue responses include
`frames`, `fps:24`, and `duration:frames/24` even after an API restart, without
requiring a local timing cache or exposing a replacement job ID. Content GET
remains read-only; the original authenticated receipt controls retrieval.
Submission performs one upstream POST with no automatic retry.

Offline integration tests exercise real FastAPI multipart requests with
captured-schema fake Comfy clients. Cross-language tests are optional:
`EASEL_MEDIA_MCP_SOURCE=/path/to/built/easel-client` also checks the production
Media MCP receipt parser and existing LTX FormData regression contracts. Live
Comfy tests require `EASEL_INTEGRATION=1` separately; offline passing tests are
not live GPU, temporal-continuity, audio-sync, or visual-review evidence.

## Evidence and rollout gate

Prior local H3 work includes October 2 FL8 T2V/I2V and two-image REF20
studies, plus an October 3 twelve-clip reference study. One original was
freshly rehashed and fully decoded October 4: 864x480, 124 frames, 24 FPS,
H.264 and AAC 32 kHz stereo. This is native recipe evidence, not this HTTP
integration, temporal guidance, identity fidelity, audio sync, or stitch
quality. The unpublished source's 491 offline tests also passed separately;
they are not counted as Easel tests.

Temporal guidance here is `graph_contract_tested` only. Review the exact cloud
head, run full tests/builds, confirm live schemas/assets, and save the receipt
for one bounded H3/continuation pilot before polling. Fully decode the media
and inspect the join; successful execution alone does not certify continuity.

```sh
uv sync --frozen --python 3.12
uv run pytest -q tests/test_h3_graph.py
uv run pytest -q
uv lock --check
uv build
git diff --check
```

Preserve `EASEL_API_KEY`, the `image-jobs` volume, and existing receipts; keep
a rollback image. Recheck PR head before promotion. New cloud commits require
a fresh review, not deployment of a moving tag.
