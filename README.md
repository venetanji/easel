# Easel

An OpenAI-compatible image and video API gateway for a separately operated
[ComfyUI](https://github.com/Comfy-Org/ComfyUI) service, with audio generation
through a separately operated Suno MCP/REST service. Python graph builders,
authenticated admission, managed uploads and durable image job receipts live here.

## Develop and run

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required:

```sh
uv sync --frozen
cp .env.example .env
# Configure the ComfyUI endpoints and authentication described in .env.example.
uv run uvicorn easel.app:app --env-file .env --host 0.0.0.0 --port 8080
```

The application reads environment variables; the command above loads `.env`
through Uvicorn. The supplied Compose configuration also reads it. Keep credentials
out of version control. API and model setup details:

- [Image jobs](IMAGE_JOBS.md)
- [Video API](VIDEO_API.md) and [video LoRAs](VIDEO_LORAS.md)
- [Video sizing, resource limits and opt-in GPU benchmarks](VIDEO_SIZING.md)
- [Model serving](MODEL_SERVING.md)
- [Audio API](AUDIO_API.md)
- [Graph provenance](GRAPH_PROVENANCE.md) and [H3 baseline](H3_BASELINE.md)

Tests use fake backends by default and do not submit GPU jobs:

```sh
uv run --frozen pytest
uv build
```

Live ComfyUI integration and cross-language client tests are explicit opt-ins;
see `tests/test_integration.py` and `tests/test_mcp_contract.py`. No model weights
or ComfyUI runtime are included in the Python package.

## Explore and use the API

Open `/docs` on your Easel origin (for the local command above,
`http://localhost:8080/docs`). Swagger groups routes into **Health**, **Models**,
**Images**, **Videos**, and **Audio**. `/openapi.json` exposes the same contract
for API tooling.

If `EASEL_API_KEY` is configured, click **Authorize** and paste the **raw key**,
without a `Bearer ` prefix. Swagger adds the bearer header to API requests,
including status and content downloads. Use Easel's key, not the upstream Suno
token. `/health` is public and only checks process liveness. With no Easel key,
API authentication is disabled; configure a key or equivalent access controls
before exposing a deployment. Documentation does not contain configured keys.

Start with `GET /v1/models`. Image model IDs are registered by the application;
video IDs appear when `COMFY_URL_VIDEO` is set, and the three audio IDs appear
when `SUNO_URL` is set. These are configuration-based listings, not guarantees
of backend readiness or installed assets. Generation can consume GPU time or
Suno credits.

### Images: submit, poll, fetch

- `POST /v1/images/generations` takes JSON: for example,
  `{"model":"qwen-image-2.1","prompt":"A watercolor lighthouse at sunrise","response_format":"url"}`.
- `/v1/images/edits` takes multipart `prompt` plus reference `image` uploads.
  `/v1/images/variations` takes an `image` upload and uses the server's variation
  instruction. Swagger includes body fields for all three routes.
- Set the optional **Prefer** header to `respond-async` for durable jobs.
  Save the HTTP 202 receipt's ID, poll its `Location` with the same bearer key,
  and wait for `completed`, `failed`, or `cancelled`. Completed `data` contains
  inline image data or authenticated, zero-based job content URLs, depending
  on `response_format`. Receipts and cached output expire after 24 hours.
- Without that header, the POST waits for output. Do not automatically repeat
  an interrupted POST: acceptance may be ambiguous and idempotency is not
  implemented. See [Image jobs](IMAGE_JOBS.md) for recovery and retention.

### Videos: discover the model's controls first

Read `GET /v1/videos/capabilities?model=ltx-2.5` or `?model=minimax-h3`, then
submit **multipart**, not JSON, to `POST /v1/videos`:

- LTX example: `model=ltx-2.5`, `prompt=A slow pan across a mountain lake`,
  `seconds=4`, `size=1280x720`.
- H3 example: `model=minimax-h3`, the same `prompt`, `frames=124`,
  `size=864x480`. **Omit `seconds` entirely** and omit LTX-only controls.

The generated Swagger form currently reflects LTX's parameters. It does not
expose all H3-only fields and can send LTX defaults that H3 rejects. Use a
multipart client for H3 and follow its model-specific capabilities and the
[Video API](VIDEO_API.md) field reference. For example, with `EASEL_URL` and
`EASEL_API_KEY` already set in your local environment:

```sh
curl "$EASEL_URL/v1/videos" \
  -H "Authorization: Bearer $EASEL_API_KEY" \
  -F 'model=minimax-h3' \
  -F 'prompt=A slow pan across a mountain lake' \
  -F 'frames=124' \
  -F 'size=864x480'
```

Save the returned video ID, poll `GET /v1/videos/{video_id}`, and retrieve
`GET /v1/videos/{video_id}/content` once completed. The queue route provides
estimates, which may be unavailable or change. All requests need the same bearer
key; video output relies on ComfyUI history rather than the durable image cache.

### Audio: latest attempt, individual tracks, saved content

`POST /v1/audio/generations` accepts model-specific JSON for `suno-music`
(musical description/lyrics), `suno-speech` (spoken script), or `suno-sound`
(sound-effect description). Swagger provides one example for each, with
`dry_run: true`: this prepares the shared browser form without clicking Create
or spending generation credits. Set it to false only when ready to generate.

Poll the returned `status_url`. It observes the **latest attempt in Suno's
shared browser**, including work from other clients, rather than a per-user job
queue. Save returned song IDs. For each track:

1. `GET /v1/audio/tracks/{song_id}` until its status is `complete`.
2. `POST /v1/audio/tracks/{song_id}/download` to save its playable audio.
3. `GET /v1/audio/tracks/{song_id}/content` to stream it, or add
   `?download=true` for a file attachment. Content returns 404 until saved.

Keep the bearer key on these requests. If generation times out or a challenge
appears, inspect shared status before retrying; a submission may already have
consumed credits. See [Audio API](AUDIO_API.md) for manual challenge recovery,
attempt abandonment, model options, and download behavior.

## License and third-party materials

Copyright (c) 2026 Venetanji. Easel-owned code, including the maintainer's Python
graph generators, is licensed under **GPL-3.0-or-later**: GNU GPL version 3 or,
at your option, any later version. The program comes without warranty. See the
full [LICENSE](LICENSE) and scoped [NOTICE](NOTICE).

Third-party material keeps its existing terms. The `licenses/` directory retains
creative-skills and official workflow-template MIT notices and ComfyUI's GPLv3
text for referenced/adapted upstream material and captured schema descriptions.
The project's SPDX field states the own-code grant; it does not relicense
upstream portions, revoke earlier MIT grants, or expand ComfyUI's permissions.
Model weights, LoRAs and other model assets have separate terms.

## Source and redistribution

The preferred application source is this repository, including `easel/`,
`pyproject.toml`, `uv.lock`, `Dockerfile`, `compose.yaml`, `scripts/`, tests,
fixtures and documentation. For a wheel/container release, record the exact
source commit and provide the corresponding source and required dependency
materials through an applicable GPL section 6 route, with clear source-access
directions alongside the binary download. A moving branch link alone does not
identify the source used to build a particular artifact.

The Dockerfile copies the README and license materials before package building
and retains application Python source. That is not, by itself, proof that every
container redistribution obligation is satisfied. Before publishing an image,
record the base-image digest, inspect installed dependency/OS notices and source
requirements, and reconcile them with the shipped layers. See the
[licensing audit](docs/license-audit.md) for the scope and verification limits.
