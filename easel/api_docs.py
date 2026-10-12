"""OpenAPI-only metadata; request parsing and validation remain in the routes."""
from __future__ import annotations

from .config import MODELS
from .qwen_graph import MAX_REFERENCES


API_DESCRIPTION = """
Generate images and videos through ComfyUI, and music, speech, or sounds through
an optional Suno service.

### Start here
1. If `EASEL_API_KEY` is configured, click **Authorize** and paste its **raw value**.
   Do not include `Bearer `: Swagger adds the `Authorization: Bearer <token>`
   header. The Suno upstream token is a different credential.
2. Call **Models → List configured models** before choosing a model. Video models
   appear when `COMFY_URL_VIDEO` is set; audio models appear when `SUNO_URL` is set.
   A listed model does not prove that its upstream assets or service are ready.
3. Expand a route, click **Try it out**, and use its request example. Generation
   can consume GPU time or Suno credits. Audio's `dry_run: true` prepares the
   shared browser form without clicking Create.

`/health` is public. Untrusted clients need the same bearer token on API routes
and output URLs when `EASEL_API_KEY` is configured. Trusted local/tailnet clients
may omit it; `EASEL_TRUSTED_CLIENT_CIDRS` controls that allowlist, and
`EASEL_REQUIRE_AUTH_CIDRS` overrides trust for public proxies. With no
`EASEL_API_KEY`, authentication is disabled. Do not publish a deployment without
setting a key or equivalent access controls. `/docs` and `/openapi.json` describe
the API; they do not reveal
configured credentials.

### Choose the workflow
- **Images:** JSON generation or multipart editing/variation. Add
  `Prefer: respond-async` to receive a durable receipt, then poll its `Location`
  and retrieve completed output. Without that header, the request waits for output.
- **Videos:** multipart submission, then status/queue polling and MP4 retrieval.
  LTX uses `seconds`; MiniMax H3 uses `frames` and rejects `seconds`. Check
  model-specific capabilities before selecting sizes or optional controls.
- **Audio:** one JSON generation route with three model-specific request shapes.
  Status observes the latest attempt in Suno's shared browser, not a private job
  queue. Save returned track IDs, wait for completion, request a saved download,
  then stream its content through Easel.

Save accepted IDs before polling. An interrupted generation POST may already have
been accepted; do not automatically resubmit it. Output URLs are authenticated,
not public share links.
""".strip()

TAGS = [
    {"name": "Health", "description": "Public process liveness; no upstream readiness check."},
    {"name": "Models", "description": "Discover model IDs enabled by this deployment's configuration."},
    {"name": "Images", "description": "Generate, edit, or vary images; poll durable async jobs and fetch output."},
    {"name": "Videos", "description": "Discover LTX/H3 controls, submit multipart jobs, poll, and download MP4 output."},
    {"name": "Audio", "description": "Generate with Suno's shared browser, recover the latest attempt, and save or stream tracks."},
]

_ASYNC_IMAGES = """
For a long-running job, set the optional `Prefer` header to `respond-async`.
Acceptance returns HTTP 202 with an `id`, `Location`, and `Retry-After: 2`.
Save the ID and poll `GET /v1/images/jobs/{job_id}` until `completed`, `failed`,
or `cancelled`. A completed job includes `data`; with `response_format: url`,
fetch its zero-based `/content/{index}` URLs with the same bearer token.
Without `Prefer`, this route waits and returns HTTP 200 with image `data`.
Do not automatically retry an interrupted POST: idempotency is not implemented.
""".strip()

_IMAGE_PROPERTIES = {
    "model": {"type": "string", "enum": list(MODELS), "default": "qwen-image-2.1", "description": "Image model ID from /v1/models."},
    "n": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1,
          "description": "Batch size. A prompt split by the configured separator produces one image per part instead."},
    "size": {"type": "string", "examples": ["1024x1024", "auto"],
             "description": "WIDTHxHEIGHT or auto. Dimensions snap to multiples of 16 and clamp to 256–1536. Generation defaults to 1024x1024; edits/variations derive omitted or auto size from the reference."},
    "response_format": {"type": "string", "enum": ["b64_json", "url"],
                        "description": "Omitting this uses EASEL_DEFAULT_RESPONSE_FORMAT (normally b64_json). URLs require authentication."},
    "server": {"type": "string", "enum": ["image", "video"], "default": "image",
               "description": "ComfyUI backend for this image job. video requires COMFY_URL_VIDEO; this does not change the selected image model."},
    "steps": {"type": "integer", "description": "Sampling steps; omitted values use the model/server default (25 for Qwen Image 2.1)."},
    "seed": {"type": "integer", "description": "Optional sampling seed; omit to choose one automatically."},
}

_PREFER_PARAMETER = {
    "name": "Prefer", "in": "header", "required": False,
    "description": "Set respond-async to submit once and poll the returned job receipt.",
    "schema": {"type": "string", "enum": ["respond-async"]},
}


def _image_body(content_type: str, properties: dict, required: list[str], **media) -> dict:
    """Describe manually parsed bodies without adding runtime validation."""
    return {
        "parameters": [_PREFER_PARAMETER],
        "requestBody": {
            "required": True,
            "content": {content_type: {
                "schema": {"type": "object", "properties": properties, "required": required},
                **media,
            }},
        },
    }


ROUTE_DOCS = {
    ("GET", "/health"): {
        "tags": ["Health"], "summary": "Check API liveness",
        "description": "Public endpoint returning `status: ok`. It does not check ComfyUI, Suno, model assets, or generation readiness.",
    },
    ("GET", "/v1/models"): {
        "tags": ["Models"], "summary": "List configured models",
        "description": "Lists registered image models, plus `ltx-2.5` and `minimax-h3` when `COMFY_URL_VIDEO` is configured, and `suno-music`, `suno-speech`, and `suno-sound` when `SUNO_URL` is configured. Discovery reflects configuration, not a successful backend health or asset check. Use the returned IDs on the matching image, video, or audio route.",
    },
    ("POST", "/v1/images/generations"): {
        "tags": ["Images"], "summary": "Generate images from a prompt",
        "description": "Send JSON with a required `prompt`. Optional model, size, batch, sampling, backend, and output-format fields are shown below. The configured prompt separator (normally `|||`) can fan out up to 16 prompts.\n\n" + _ASYNC_IMAGES,
        "openapi_extra": _image_body("application/json", {
            "prompt": {"type": "string", "minLength": 1, "description": "Describe the image to generate."},
            **_IMAGE_PROPERTIES,
        }, ["prompt"], example={
            "model": "qwen-image-2.1", "prompt": "A watercolor lighthouse at sunrise",
            "size": "1024x1024", "n": 1, "response_format": "url",
        }),
    },
    ("POST", "/v1/images/edits"): {
        "tags": ["Images"], "summary": "Edit images with instructions and references",
        "description": f"Send multipart form data with a `prompt` and 1–{MAX_REFERENCES} Qwen reference files in repeated `image` fields (`image[]` is also accepted). Describe the desired change or how to combine references. Omit size or use `auto` to derive it from the reference. This is instruction/reference editing; uploaded masks are rejected.\n\n" + _ASYNC_IMAGES,
        "openapi_extra": _image_body("multipart/form-data", {
            "prompt": {"type": "string", "description": "Editing instruction.", "example": "Turn the scene into a watercolor painting"},
            "image": {"type": "array", "minItems": 1, "maxItems": MAX_REFERENCES,
                      "items": {"type": "string", "format": "binary"},
                      "description": "Reference image uploads. Repeat this field for multiple references; image[] is an accepted alias."},
            **_IMAGE_PROPERTIES,
        }, ["prompt", "image"]),
    },
    ("POST", "/v1/images/variations"): {
        "tags": ["Images"], "summary": "Create variations of an image",
        "description": "Send a multipart `image` upload (`image[]` is also accepted). Only the first supplied image is used. Easel uses its configured variation instruction; a caller-supplied `prompt` is not used. Omit size or use `auto` to derive it from the reference.\n\n" + _ASYNC_IMAGES,
        "openapi_extra": _image_body("multipart/form-data", {
            "image": {"type": "string", "format": "binary", "description": "Source image for the variation."},
            **_IMAGE_PROPERTIES,
        }, ["image"]),
    },
    ("GET", "/v1/images/jobs/{job_id}"): {
        "tags": ["Images"], "summary": "Poll an async image job",
        "description": "Use the ID or Location returned by an image POST with `Prefer: respond-async`. States are `queued`, `in_progress`, `completed`, `failed`, or `cancelled`. Completed responses include image `data`. Polling does not submit generation again, and the chosen backend is stored with the receipt. Receipts and cached output expire 24 hours after creation (HTTP 410); unknown IDs return 404. Queue estimates may be null and are not guaranteed completion times.",
    },
    ("GET", "/v1/images/jobs/{job_id}/content/{index}"): {
        "tags": ["Images"], "summary": "Fetch a completed image job output",
        "description": "Retrieve cached PNG bytes from a completed async image job. `index` is zero-based; use the URL in completed job `data` when response_format was url. Requires the same bearer token as submission. Unfinished jobs return 409 and missing output indices return 404. Cached output expires with the receipt after 24 hours.",
    },
    ("GET", "/v1/images/view"): {
        "tags": ["Images"], "summary": "Fetch a synchronous image result",
        "description": "Follow the authenticated URL returned by a synchronous image request with `response_format: url`. Keep the returned `filename`, `subfolder`, `type`, and optional `server` query values; this reads the selected ComfyUI backend rather than the durable async cache. `type` accepts output, input, or temp. For async jobs, use their job content URLs instead.",
    },
    ("GET", "/v1/videos/loras"): {
        "tags": ["Videos"], "summary": "List model-specific video adapters",
        "description": "Defaults to `model=ltx-2.5`: lists curated LoRAs, required fields, support/validation labels, and live installed state. Installed does not mean workflow-validated. `model=minimax-h3` returns an empty list because H3 uses a fixed curated recipe; arbitrary adapter paths are not accepted. Requires a configured video backend.",
    },
    ("GET", "/v1/videos/capabilities"): {
        "tags": ["Videos"], "summary": "Inspect model-specific video capabilities",
        "description": "Read this before submitting video. Set `model=ltx-2.5` (default) or `model=minimax-h3` to discover size/duration budgets, seed bounds, upload limits, and model-specific controls. LTX supports seconds and timed guiding frames; H3 uses the 17k+5 frame grid and semantic/temporal reference metadata. Runtime availability and code support are separate; discovery does not generate video or prove visual quality.",
    },
    ("POST", "/v1/videos"): {
        "tags": ["Videos"], "summary": "Submit an LTX or H3 video job",
        "description": """Send **multipart/form-data**, not JSON. Required fields are `model` and `prompt`.

- **LTX:** `model=ltx-2.5`, `prompt=A slow pan across a mountain lake`,
  `seconds=4`, `size=1280x720`. Optional `input_reference` accepts PNG/JPEG/WebP.
  Discover LoRAs and guiding-frame controls using capabilities and the adapter catalog.
- **H3:** `model=minimax-h3`, `prompt=A slow pan across a mountain lake`,
  `frames=124`, `size=864x480`. H3 accepts PNG/JPEG stills and its own semantic
  references/temporal groups. **Omit `seconds` entirely**, even the LTX default.
  H3 rejects LTX-specific controls. Its duration is `frames / 24` seconds.

**Swagger form limitation:** the automatically generated form below describes
the LTX function parameters. It does not expose all H3-only fields and may send
LTX defaults that H3 rejects. For H3, use the multipart example above with a client
that omits `seconds`; do not treat the displayed form as H3's complete schema.
Use `/v1/videos/capabilities?model=minimax-h3` for H3 constraints.

The returned `id` identifies an accepted, queued job; save it, poll
`GET /v1/videos/{video_id}`, optionally inspect `/v1/videos/queue/{video_id}`,
and fetch `/v1/videos/{video_id}/content` after status becomes `completed`.
Sizes and durations are subject to the configured resource budget. A listed
model does not guarantee installed assets or compatible native nodes. An
ambiguous POST failure must not be automatically retried.""",
    },
    ("GET", "/v1/videos/{video_id}"): {
        "tags": ["Videos"], "summary": "Poll a video job",
        "description": "Use the `id` returned by video submission. States are queued, in_progress, completed, failed, or cancelled. When completed, fetch `/v1/videos/{video_id}/content`. Video status and output depend on ComfyUI queue/history retention; they are separate from Easel's durable image cache. Polling does not submit a replacement job.",
    },
    ("GET", "/v1/videos/queue/{video_id}"): {
        "tags": ["Videos"], "summary": "Inspect video queue position and estimates",
        "description": "Returns one video's status, queue position, jobs ahead, queue counts, and completion estimates. Position 0 means running; pending positions start at 1. Estimates can be null until successful runtime samples exist, include generation time, and are not promises. Use normal video status/content routes for completion and download.",
    },
    ("GET", "/v1/videos/{video_id}/content"): {
        "tags": ["Videos"], "summary": "Download a completed video",
        "description": "Returns MP4 bytes with attachment disposition after the job is completed. Keep the bearer token on this request. Unfinished or failed generation returns 409 (video_not_ready); unknown or expired IDs cannot be retrieved. Output remains backed by ComfyUI, not the durable image cache.",
    },
    ("POST", "/v1/audio/generations"): {
        "tags": ["Audio"], "summary": "Generate music, speech, or a sound",
        "description": """Send JSON matching one of three model-specific shapes:
- `suno-music`: prompt is a musical style/description; optionally supply lyrics,
  title, or music controls. Music accepts a prompt, lyrics, or both. Instrumental
  requests require empty lyrics. Optional `suno_model` is an exact native ID from
  `/v1/audio/models`, not the Easel routing ID; omitted or blank uses the backend's
  configured default.
- `suno-speech`: prompt is the script to speak; optional tone/background music.
- `suno-sound`: prompt describes a sound; optional one_shot/loop and bpm.

All accept `dry_run: true` to prepare the shared browser form without clicking
Create or spending generation credits. The examples below use this safer first
step; set false only when ready to generate. Unknown fields are rejected.

An accepted pending/submitted/CAPTCHA attempt returns HTTP 202, status_url,
Location, and Retry-After: 5. Poll status_url rather than submitting again. Solve
any challenge manually through the returned novnc_url, then continue polling.
Status is the latest shared-browser attempt, not a per-user job receipt. Save
song IDs, inspect each track until complete, POST its /download to save audio,
then GET its /content. After a timeout/disconnect, check status and the Suno
library before retrying: generation may already have consumed credits.""",
        "openapi_extra": {"requestBody": {"content": {"application/json": {"examples": {
            "music": {"summary": "Prepare instrumental music", "value": {"model": "suno-music", "prompt": "Gentle ambient piano with warm strings", "make_instrumental": True, "dry_run": True}},
            "speech": {"summary": "Prepare a spoken script", "value": {"model": "suno-speech", "prompt": "Welcome to the exhibition.", "tone": "Warm and calm", "dry_run": True}},
            "sound": {"summary": "Prepare a sound effect", "value": {"model": "suno-sound", "prompt": "A wooden door creaks open", "sound_type": "one_shot", "dry_run": True}},
        }}}}},
    },
    ("GET", "/v1/audio/generations/status"): {
        "tags": ["Audio"], "summary": "Inspect the latest shared audio attempt",
        "description": "Observes the latest attempt in Suno's shared browser, including work started by MCP or direct REST clients. It is not per-user status or a persistent Easel job queue; a later attempt replaces it and Suno restarts lose attempt state. Save track IDs for individual polling. Status can report pending, submitted, captcha_required, complete, error, abandoned, or idle. Poll this after an ambiguous submission before deciding to retry; do not submit another generation to check progress.",
    },
    ("POST", "/v1/audio/generations/abandon"): {
        "tags": ["Audio"], "summary": "Close an unconfirmed audio attempt",
        "description": "Only after manually checking a stuck or cancelled challenge, send the exact `attempt_id` from shared generation status. This closes the browser's unconfirmed wait. It does not cancel accepted tracks or refund generation credits. Do not use it as routine polling or apply it to an unrelated attempt.",
    },
    ("GET", "/v1/audio/tracks/{song_id}"): {
        "tags": ["Audio"], "summary": "Check an individual audio track",
        "description": "Use a song UUID returned by generation/status. Wait for `status: complete`, then POST the track's `/download` route to save playable audio before using `/content`. Returned audio_url and download_url point to authenticated Easel routes; POST is required for download_url. Track IDs remain useful after a later generation replaces shared attempt status.",
    },
    ("POST", "/v1/audio/tracks/{song_id}/download"): {
        "tags": ["Audio"], "summary": "Save a completed audio track",
        "description": "Ask Suno to save this track's playable audio in its existing download volume. Wait for the track to complete first. The response supplies an authenticated Easel audio_url; GET that URL to retrieve the saved bytes. This is the authenticated-player download, not paid WAV/stem export, and retains the actual file format rather than assuming MP3.",
    },
    ("GET", "/v1/audio/tracks/{song_id}/content"): {
        "tags": ["Audio"], "summary": "Stream or download saved audio",
        "description": "GET the saved file through Easel with the same bearer token. First POST `/v1/audio/tracks/{song_id}/download`; content returns 404 until audio is saved. Set `download=true` for attachment disposition. **In Swagger:** authorize first, set the download query parameter to true, click **Execute**, then click **Download file** in the response. Open the downloaded file in your audio player. The inline player can fail on bearer-protected deployments because it makes a separate media request without Swagger's Authorization header; use the download workflow instead. Never put a bearer token in the URL. Range and conditional request headers are forwarded for playback/caching, and the actual file type is preserved. Audio stays in Suno's download volume, separate from Easel's image cache.",
    },
    ("GET", "/v1/audio/models"): {
        "tags": ["Audio"], "summary": "List native Suno music models",
        "description": "Discovers the logged-in account's base and custom music models. Use an exact returned ID as suno_model on suno-music requests; omitted or blank uses the backend's configured default_model, not its previous selection. The response also includes selected_model and data entries with availability, selection, and default flags. Discovery does not change the selected model or spend generation credits, but shares Suno's browser queue and pending-attempt guard. Unlike /v1/models, this route contacts the backend.",
    },
}


_REQUEST_SCHEMA_NAMES = {
    "/v1/images/generations": ("application/json", "ImageGenerationRequest"),
    "/v1/images/edits": ("multipart/form-data", "ImageEditRequest"),
    "/v1/images/variations": ("multipart/form-data", "ImageVariationRequest"),
    "/v1/videos": ("multipart/form-data", "VideoGenerationRequest"),
    "/v1/audio/generations": ("application/json", "AudioGenerationRequest"),
    "/v1/audio/generations/abandon": ("application/json", "AudioGenerationAbandonRequest"),
}

_COMPONENT_SCHEMA_NAMES = {
    "MusicGeneration": "AudioMusicGenerationRequest",
    "SpeechGeneration": "AudioSpeechGenerationRequest",
    "SoundGeneration": "AudioSoundGenerationRequest",
    "HTTPValidationError": "ValidationErrorResponse",
    "ValidationError": "ValidationErrorDetail",
}


def _replace_schema_refs(value, replacements: dict[str, str]) -> None:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference in replacements:
            value["$ref"] = replacements[reference]
        for nested in value.values():
            _replace_schema_refs(nested, replacements)
    elif isinstance(value, list):
        for nested in value:
            _replace_schema_refs(nested, replacements)


def configure_openapi(app) -> None:
    """Name documentation schemas without altering runtime request models."""
    generate_openapi = app.openapi

    def openapi() -> dict:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = generate_openapi()
        components = schema["components"]["schemas"]
        names = dict(_COMPONENT_SCHEMA_NAMES)
        prefix = "#/components/schemas/"
        for endpoint, (content_type, name) in _REQUEST_SCHEMA_NAMES.items():
            media = schema["paths"][endpoint]["post"]["requestBody"]["content"][content_type]
            body = media["schema"]
            if "$ref" in body:
                names[body["$ref"].removeprefix(prefix)] = name
            else:
                components[name] = body
            media["schema"] = {"$ref": prefix + name}
        for original, name in names.items():
            if original in components:
                components[name] = components.pop(original)
        for name, definition in components.items():
            definition["title"] = name
        _replace_schema_refs(schema, {prefix + original: prefix + name
                                      for original, name in names.items()})
        schema["components"]["schemas"] = dict(sorted(components.items()))
        app.openapi_schema = schema
        return schema

    app.openapi = openapi
