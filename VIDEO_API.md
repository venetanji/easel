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
| `size` | no | `1280x720`, `720x1280`, `1792x1024`, or `1024x1792` (default `1280x720`) |
| `input_reference` | no | PNG, JPEG, or WebP image for image-to-video generation |

Fractional durations are not accepted. At 24 FPS, the workflow uses
`seconds * 24 + 1` frames for LTX frame alignment, so a clip is approximately
1/24 second longer than the requested duration.

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
  `in_progress`, `completed`, or `failed` status.
- `GET /v1/videos/{video_id}/content` returns the completed MP4 bytes.
- `GET /v1/videos/queue/{video_id}` returns that job's queue position, jobs ahead,
  queue counts, and an estimated remaining time to completion. Estimates use the
  average duration of the last 20 completed videos observed by Easel and remain
  `null` until Easel has timing data. Position `0` means running; queued positions
  start at `1`.

Requests use the same optional `Authorization: Bearer <EASEL_API_KEY>` header as
the image endpoints. ComfyUI history must retain the prompt until the caller
downloads its result.
