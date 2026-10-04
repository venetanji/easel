# Video API

LiteLLM's [`/videos` endpoint documentation](https://docs.litellm.ai/docs/videos)
uses the OpenAI Video Generation API specification and lists OpenAI, Azure,
Gemini, Vertex AI, and Runway as supported providers. Easel implements that
request shape and submits the work to the configured ComfyUI video server,
using LTX-2.5 or native MiniMax H3 workflows. H3's `frames`, semantic-reference
and temporal-group fields are Easel model-specific extensions to that shape;
they are not shared LTX controls. Registration describes server code support,
not proof of deployment, live runtime compatibility or visual validation.

Set `COMFY_URL_VIDEO` to the ComfyUI video server. The Docker Compose default
points to `comfy-docker-tailscale-serve-video-1` on `tailscale-mesh`.

## Create an LTX-2.5 video

`POST /v1/videos` accepts `multipart/form-data`:

| Field | Required | Values |
|---|---|---|
| `model` | yes | `ltx-2.5` |
| `prompt` | yes | Text describing the clip |
| `seconds` | no | Any whole number from `1` through `12` (default `4`) |
| `size` | no | `512x320`, `640x384`, `768x512`, `1280x720`, `720x1280`, `1792x1024`, or `1024x1792` (default `1280x720`) |
| `input_reference` | no | PNG, JPEG, or WebP image for image-to-video generation |
| `camera_lora` | no | `dolly-in`, `dolly-out`, `dolly-left`, `dolly-right`, `jib-up`, `jib-down`, `static` |
| `camera_lora_strength` | no | Finite `0` through `2`, default `0.8`; requires `camera_lora` |
| `loras` | no | JSON string of registered `id` / optional `strength` objects; at most four adapters including `camera_lora` |
| `motion_speed` | no | `0.025` through `1`, required with the `slow-motion` LoRA |
| `seed` | no | Integer `0` through `2**64-2`; both sampling-pass seeds must fit unsigned 64 bits |
| `lora_reference` | no | PNG, JPEG, or WebP reference sheet, required with `ingredients` |
| `lora_reference_strength` | no | Finite `0` through `1`, default `1`; requires `ingredients` |
| `guiding_frames` | no | JSON array of 1–8 `{image_index, frame_index, strength?}` entries |
| `guiding_images` | no | Repeated still PNG/JPEG/WebP uploads, one per guiding-frame entry |

Fractional durations are not accepted. At 24 FPS, the workflow uses
`seconds * 24 + 1` frames for LTX frame alignment, so a clip is approximately
1/24 second longer than the requested duration.

The generation graph samples at half resolution (8 steps), applies a latent
2x spatial upscale, then refines (3 steps). Camera/regular LoRA patches feed
both passes. The nominal `1280x720` and `720x1280` presets currently decode to
`1280x704` and `704x1280`: the coarse latent grid floors dimensions to multiples
of 32. The other presets are aligned. Low presets are useful for bounded tests,
not evidence of delivery-resolution quality or high-resolution VRAM capacity.

## Create a MiniMax H3 video

The same `POST /v1/videos` endpoint accepts these H3 multipart fields:

| Field | Required | Values |
|---|---|---|
| `model` | yes | `minimax-h3` |
| `prompt` | yes | Nonblank text describing the clip |
| `frames` | no | Exactly `17k+5` in `124..362`, default `124` |
| `size` | no | Only `864x480`, also the default |
| `seed` | no | Unsigned decimal integer `0..18446744073709551615`, default `0`; JavaScript must send an exact decimal string |
| `input_reference` | no | One PNG/JPEG still for FL8 first-frame image to video |
| `images` | no | Repeated PNG/JPEG still uploads indexed by metadata in this request |
| `semantic_references` | no | JSON array of one or two `{"image_index":0}` objects referring to `images` |
| `temporal_groups` | no | JSON array of one through three `{"frame_index":0,"image_indices":[0,1,2,3,4]}` objects referring to `images` |

Omit conditioning for FL8 text to video. `input_reference` selects FL8 image to
video and cannot be combined with `images`, semantic references or temporal
groups. Semantic Picture references select REF20; they preserve declared order
and are not timeline anchors. Experimental temporal groups also select REF20
and can coexist with separately indexed semantic references. The two profiles
retain their fixed 8-step and 20-step recipes respectively; clients cannot
select alternate samplers, assets or adapters.

Each temporal group contains exactly 1, 5, 22 or 39 stills in `image_indices`
order, fits entirely within the output frame count, and must not overlap another
group. `frame_index` is a nonnegative pixel-frame position, not seconds or a
latent index. Each `images` index must identify a supplied upload and be used
exactly once across all metadata. Orphans, duplicates, unknown/empty metadata,
repeated singleton fields and strengths are rejected before upstream work.
There are at most 119 still uploads. Each must safely decode as PNG/JPEG bytes
matching its MIME, with at most 32 million pixels each and 64 million pixels
combined; combined bytes are capped at 32 MiB. Each temporal group's decoded
dimensions must match after EXIF orientation. Client filenames are discarded
in favor of generated managed basenames; paths, URLs and arbitrary graphs
are not accepted.

H3 always outputs 24 FPS. Duration is exactly `frames/24` (the default is
`124/24`, approximately 5.166667 seconds). Explicit `seconds` is rejected,
including an explicitly supplied LTX default; H3 never silently maps or rounds
LTX durations. H3 also rejects WebP, LTX camera/Ingredients/motion controls,
source-audio guidance, control video, guide strengths, negative/CFG controls
and custom sampling parameters.

Authenticated `GET /v1/videos/capabilities?model=minimax-h3` describes the
frame grid, size/seed bounds, typed metadata and upload limits. Each profile
reports code `supported` separately from native node/curated-asset `available`.
Discovery is read-only and never loads models, uploads files or submits a
prompt. Selected-profile schema/asset checks run again before H3 queue
admission or uploads. These are fixed code-level capability labels:
`graph_contract_tested`, `gpu_executed:false`, and `visually_reviewed:false`,
not live per-deployment pilot results. Discovery does not certify execution,
continuity, identity fidelity or audio/stitch quality.
See [H3_BASELINE.md](H3_BASELINE.md) for native profile details and the rollout
gate.

## Versioned capabilities and timed image guides

Authenticated `GET /v1/videos/capabilities` defaults to LTX-2.5; an explicit
`?model=ltx-2.5` selects the same contract and unknown models are rejected.
The LTX `video.capabilities` object includes `schema_version: 1`, model, FPS, accepted durations/sizes, exact seed
bounds, adapter bounds, upload limits and guiding-frame support. Runtime
`guiding_frames.available` requires compatible `LTXVAddGuide` and
`LTXVCropGuides` input/output schemas. Discovery performs no generation or
uploads. A backend outage is an error, not proof that guides are unsupported.

The guide metadata and files belong to the same multipart request:

```text
guiding_frames=[{"image_index":0,"frame_index":1,"strength":0.7},{"image_index":1,"frame_index":48,"strength":0.5}]
guiding_images=@first.png
guiding_images=@last.png
```

For a two-second video, pixel-frame positions run from 0 through 48 inclusive.
Positions are integers, not latent-frame indices or seconds. There is no
multiple-of-eight restriction for single-image guides. Every image index must
identify exactly one corresponding upload; frame positions must be unique.
Easel sorts anchors by frame position while retaining their image association.
Finite strengths are 0–1, default 1, and retain the same value in both passes.
Guidance is soft conditioning: even strength 1 does not promise a pixel-perfect
endpoint, exact motion, seamless boundary or loop.

All reference uploads share a 32 MiB combined limit. Timed guides additionally
must decode as one still image, with at most 32 megapixels; animated PNG/WebP
and malformed/mislabeled files are rejected. Guide mode cannot be combined
with `input_reference`, Ingredients or `lora_reference`. Camera/regular LoRAs
can be used where their existing requirements allow it. Cinemagraph and
slow-motion still require `input_reference`, so cannot accompany timed guides.
Unknown fields, repeated singleton fields, unpaired uploads and invalid options
are rejected before backend upload or submission instead of being ignored.

Guides enter the **video-only** latent before audio concatenation. Guide tails
and conditioning metadata are cropped before the 2x spatial upscale. Guides
are then reapplied to the cleaned conditioning/upsampled latent and cropped
again before decode. Existing model identities, 8+3 schedules, seed/seed+1 and
generated audio remain unchanged. Full control-video/IC adapters are separate
workflows and are not enabled by this path.

The new two-pass temporal guide path is `graph_contract_tested`, backed by the
[official LTX-2.5 first/last blueprint](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/blueprints/First%20%26%20Last%20Frame%20to%20Video%20(LTX-2.5).json)
and [node semantics](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/comfy_extras/nodes_lt.py#L251-L519).
It has not been GPU/visually verified in this change. The official blueprint is
single-stage; composing its guide/crop boundary across Easel's two stages is
covered by graph contract tests, not claimed as an official two-stage recipe.
Use an explicitly authorized bounded pilot before scaling. See
[graph provenance](GRAPH_PROVENANCE.md) for the current reuse boundary.

## Adapter discovery and requirements

Authenticated `GET /v1/videos/loras` defaults to the LTX catalog; explicit
`?model=ltx-2.5` returns the same contract and unknown models are rejected.
`?model=minimax-h3` returns `{"object":"list","model":"minimax-h3","data":[]}`
without probing LTX adapters: H3's curated FL8 adapter is fixed by its recipe.
The LTX catalog returns the curated identities, pinned
filenames/revisions/checksums, required input fields, `supported`, `validation`,
and live backend `installed` state. Installation is not workflow validation.
Unregistered paths and duplicate IDs are rejected. Installed but unsupported
IC adapters remain discoverable and are rejected before uploads/submission. JavaScript consumers must send `seed` as an exact
decimal string; a large JavaScript number can silently lose precision.
Missing selected assets return HTTP 503 (`lora_not_installed`).

Example form fields:

```text
camera_lora=dolly-in
camera_lora_strength=0.8
loras=[{"id":"camera-static","strength":0.8}]
```

Use either the shorthand or the JSON entry for a given adapter, not both.

`cinemagraph` requires `input_reference`; Easel adds `CINEMAGRAPH_MOTION` to the
prompt. It rejects combinations with moving camera adapters. `slow-motion`
requires `input_reference` and `motion_speed`: at `0.2`, motion conditioning
uses 120 FPS while playback/audio stay at 24 FPS. It is generated slow motion,
not deterministic interpolation of an uploaded clip.

`ingredients` requires a separate `lora_reference` sheet and at least 5 seconds
(121 looped reference frames). Prepare the sheet at the output aspect ratio.
Its graph uses the IC loader's metadata, adds guides in both passes and crops
guide frames before upscale/decode. It cannot yet be stacked with other LoRAs.
Required IC nodes are checked before upload. These non-camera paths are
experimental; consult `VIDEO_LORAS.md` for validation evidence and boundaries.

The create response is returned as soon as ComfyUI accepts the prompt:

```json
{
  "id": "video_...",
  "object": "video",
  "created_at": 1790000000,
  "status": "queued",
  "completed_at": null,
  "expires_at": 1790086400,
  "error": null,
  "model": "ltx-2.5",
  "progress": 0
}
```

The `video_...` ID carries the ComfyUI prompt ID, so Easel can read status and
output references from ComfyUI's queue and history after an Easel restart.
New H3 IDs add optional `:frames` to the encoded-model component, preserving
the existing receipt and published Media MCP ID character contracts. H3
creation, polling and queue responses include exact `frames`, `fps:24`, and
`duration:frames/24`, including after an API restart. Existing LTX IDs and
response shapes are unchanged. Content retrieval uses the same authenticated
receipt and performs no replacement submission.

## Check and download

- `GET /v1/videos/{video_id}` returns the same video object with `queued`,
  `in_progress`, `completed`, `failed`, or backend-interrupted `cancelled` status.
- `GET /v1/videos/{video_id}/content` returns the completed MP4 bytes.
- `GET /v1/videos/queue/{video_id}` returns that job's queue position, jobs ahead,
  queue counts, and an estimated remaining time to completion. Estimates use the
  average duration of the last 20 completed videos observed by Easel and remain
  `null` until Easel has timing data. Position `0` means running; queued positions
  start at `1`.

The queue response includes `created_at`, `expires_at`, `completed_at`, `error`
and `progress`. `estimated_wait_seconds` is time until completion, including
generation, and `estimated_completion_at` is Unix seconds. Running estimates
decrease from the execution/observed start when timing samples exist. Video
averages are process-local and become cold/null after an Easel restart. Only
terminal successful history counts as completion; previews alone do not.

Requests use the same optional `Authorization: Bearer <EASEL_API_KEY>` header as
the image endpoints. ComfyUI history must retain the prompt until the caller
downloads its result.

Video retention is 24 hours from creation. Easel restarts preserve the ID, but
ComfyUI must retain queue/history/output; a ComfyUI restart/history purge can
make a video unavailable. Expired/unknown video IDs return HTTP 404. There is
no video cancellation or idempotency/recovery endpoint. Polling never submits
a replacement job. Do not treat receipt loss as permission to retry blindly.

For queued image generation, edits and variations, see `IMAGE_JOBS.md`.

## Offline cross-repository checks

The default server `pytest` suite uses fake ComfyUI backends. The extra
cross-language contract suite captures actual Media MCP `FormData`, replays
those exact parts through FastAPI, and asserts guide/LoRA/seed graph inputs.
It also parses the real server discovery/catalog through Media MCP. Build the
client package first, then run from the server checkout:

```bash
EASEL_MEDIA_MCP_SOURCE=/path/to/easel-client python -m pytest -q tests/test_mcp_contract.py
```

The companion CLI has an in-process `contract/server_contract.py` suite using
`EASEL_SERVER_SOURCE=/path/to/easel`. Neither contract suite starts real GPU
work or requires an API key. The client's separate live runner defaults to help;
actual camera/guided pilots require a local key and explicit cost opt-in. Keep
accepted receipts and use resume, never replacement submissions after a timeout.
