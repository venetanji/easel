# Video API

LiteLLM's [`/videos` endpoint documentation](https://docs.litellm.ai/docs/videos)
uses the OpenAI Video Generation API specification and lists OpenAI, Azure,
Gemini, Vertex AI, and Runway as supported providers. Easel implements that
request shape and submits the work to the configured ComfyUI video server,
using the LTX-2.5 workflow.

Set `COMFY_URL_VIDEO` to the ComfyUI video server. The Docker Compose default
points to `comfy-docker-tailscale-serve-video-1` on `tailscale-mesh`.

## Create a video

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

Fractional durations are not accepted. At 24 FPS, the workflow uses
`seconds * 24 + 1` frames for LTX frame alignment, so a clip is approximately
1/24 second longer than the requested duration.

The generation graph samples at half resolution (8 steps), applies a latent
2x spatial upscale, then refines (3 steps). Camera/regular LoRA patches feed
both passes. The nominal `1280x720` and `720x1280` presets currently decode to
`1280x704` and `704x1280`: the coarse latent grid floors dimensions to multiples
of 32. The other presets are aligned. Low presets are useful for bounded tests,
not evidence of delivery-resolution quality or high-resolution VRAM capacity.

## Adapter discovery and requirements

Authenticated `GET /v1/videos/loras` returns the curated identities, pinned
filenames/revisions/checksums, required input fields, `supported`, `validation`,
and live backend `installed` state. Installation is not workflow validation.
Unregistered paths and duplicate IDs are rejected. Installed but unsupported
IC adapters remain discoverable and are rejected before uploads/submission.
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
