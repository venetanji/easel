# Model Serving and Queue Admission

This server revision advertises `qwen-image-2.1`, `ltx-2.5`, and `minimax-h3`
when `COMFY_URL_VIDEO` is configured. Image requests with an omitted/blank model
use Qwen; video requests require an explicit `ltx-2.5` or `minimax-h3` model.
This describes code support, not independently verified deployment state.
`flux2-9b`, `flux2-4b` and `flux-2.5` are not accepted for new generation, editing
or variation requests. Adapter choices do not create additional base models.
If `COMFY_URL_VIDEO` is not configured, Qwen is the only ComfyUI model advertised.
Setting `SUNO_URL` independently adds `suno-music`, `suno-speech`, and `suno-sound`.
These audio routing IDs use the separate Suno REST/browser backend, not ComfyUI;
see [AUDIO_API.md](AUDIO_API.md) for options, status polling and saved audio.

Qwen uses its matching INT8 transformer/text encoder, BF16 VAE and 25 sampling
steps unless a request supplies `steps`. The old global eight-step Flux default
does not override Qwen's model-specific default. LTX retains its distilled 8+3
sampling passes and existing video/adapter contracts.

Qwen edits and variations preserve reference alpha by rejoining `LoadImage`'s
RGB and mask outputs with the standard ComfyUI `JoinImageWithAlpha` node before
`TextEncodeQwenImage21`. The join node handles the inverse-alpha mask convention;
opaque references retain an alpha of one. The Qwen node composites RGBA over
white for vision conditioning and retains all four channels for its VAE input.
The image backend must include `JoinImageWithAlpha`. This preserves reference
conditioning, not an exact output mask: transparent output still depends on the
prompt and must be verified in the generated PNG.

H3 uses native FL8 for text/first-frame image generation and REF20 for semantic
Picture references or experimental temporal groups. Its output default is
864x480 with exactly 124 frames at 24 FPS (124/24 seconds); accepted frame counts
are exactly 17k+5 in 124..362. It accepts the full unsigned 64-bit seed range
(default 0) without LTX's second-pass increment. Explicit `seconds`, source
audio and LTX adapter/motion controls are unsupported for H3. H3 multipart
reference fields and durable timing are documented in [VIDEO_API.md](VIDEO_API.md).

Authenticated `GET /v1/videos/capabilities?model=minimax-h3` reports supported
profiles separately from read-only native schema/curated-asset availability;
it does not load models or certify GPU execution or visual quality. Omit
`model` for existing LTX capabilities; unknown models are rejected. H3
admission checks the selected runtime profile before queue work or uploads.
`GET /v1/videos/loras?model=minimax-h3` returns an empty catalog because H3's
FL8 adapter is fixed by its recipe. See [H3_BASELINE.md](H3_BASELINE.md) for
profile contracts, offline evidence and the local rollout gate.

## Backend placement

`COMFY_URL_IMAGE` configures the primary image backend. It takes precedence over
the legacy `COMFY_URL_FLUX`; unset/blank values fall through to that legacy
variable and then the existing primary-backend default. Docker Compose applies
the same precedence. `COMFY_URL_VIDEO` remains the video backend.

Keep these URLs distinct when backed by different GPUs, allowing Qwen images
and LTX/H3 videos to execute independently rather than paying model-switching costs
on one device. The existing `server=video` image override remains compatible;
it is not required for default image requests.

The inspected Thor deployment uses the 12 GiB RTX 3080 Ti for primary images and
the 24 GiB RTX 3090 for video. A single 512x512, 25-step Qwen image completed
through the primary async API in about six seconds during the retirement check.
That pilot establishes a viable default route, not a throughput guarantee or
evidence that every size/batch fits VRAM. Do not reroute both models onto the
video GPU solely because Qwen's weights are large.

If both URLs identify the same backend (ignoring a trailing slash), Easel reuses
its client, submission lock and synchronous limiter. Use the same canonical URL
for aliases of one queue; unrelated hostnames pointing at the same physical
process cannot be inferred automatically. A single image watcher cycle fetches
one queue snapshot per backend client, even when both logical routes use it.

## Queue behavior

Continue using ComfyUI's FIFO queue and the existing async receipt contracts:

- Image generation/edit/variation with `Prefer: respond-async` receives a durable
  HTTP 202 receipt before execution completes. Accepted jobs do not occupy the
  synchronous `COMFY_MAX_INFLIGHT` slot; the cache worker observes and saves
  terminal output after client disconnect.
- Video submission returns its accepted ID without waiting for sampling.
  New H3 receipts also preserve exact `frames`, `fps`, and `duration` across
  API restarts in their existing encoded job ID; legacy video IDs remain valid.
- Admission checks include running/pending external ComfyUI jobs. At eight
  upstream jobs, new admission returns HTTP 429 with `Retry-After`. Async image
  reservations retain their existing SQLite-backed limit as well.
- Queue-check/upload/submit sequences are serialized per backend, preventing
  concurrent Easel image/video requests from racing past the same queue snapshot.
  The lock is released after submission, not held through GPU execution or
  content download. Different GPU backends use different locks.
- The synchronous limiter is also shared when routes alias one backend. A caller
  cannot gain another synchronous slot simply by selecting the other route.

Run **one Easel admission worker**, as the supplied Compose/uvicorn command does.
Submission locks coordinate that process; they are not a distributed scheduler
across multiple Easel replicas. Image reservations are durable/atomic across
workers, but video and synchronous admission still require a single owner.
External direct ComfyUI submitters do not participate in Easel's lock and can
change queue depth after a snapshot. Do not scale replicas or GPU concurrency
without a separately coordinated admission design.

The v0.0.2 client owns monitoring and downloading after acceptance. Keep IDs and
their original endpoint/model; do not poll repeatedly with agent tools or retry
an ambiguous generation POST. Queue ETA remains approximate, and represents
time until completion, including execution, not queue waiting alone.

## Retirement and recovery

Retirement rejects new Flux requests only. Existing queued Flux graphs can
finish, and their image receipts/cache/content remain retrievable until the
normal 24-hour expiry. Retrieval does not re-resolve the retired model or submit
a substitute job. Do not remove old weights while accepted graphs need them.

Preserve `EASEL_JOB_DIR` and Compose's `image-jobs` volume across redeployment.
Completed cached images survive backend history loss; unfinished images and
videos still depend on the upstream queue/history/output. An Easel restart does
not clear ComfyUI's queue. No model weights, history or cached output must be
deleted to retire discovery/admission.

See `IMAGE_JOBS.md`, `VIDEO_API.md` and `VIDEO_LORAS.md` for the existing receipt,
retention and adapter boundaries. `|||` still permits up to 16 independent image
prompts in one graph and ignores `n` when more than one nonempty segment is
present. The v0.0.2 client currently parses at most four returned images; its
larger-fan-out fix and advanced MCP fields remain client-session work.

## Validation

`tests/test_model_serving.py` checks exact model discovery, retired-ID rejection,
retrieval of pending/cached retired jobs, simultaneous mixed admission, shared
alias capacity, independent GPU queues and releasing admission before execution.
Existing graph/adapter/cache tests remain in place. Legacy Flux graph-builder
tests are retained as historical code coverage; those builders are no longer
used by active Easel endpoints.
