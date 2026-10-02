# Model Serving and Queue Admission

The deployed service exposes only `qwen-image-2.1` and `ltx-2.5`. Image requests
with an omitted/blank model use Qwen; video requests require `ltx-2.5` explicitly.
`flux2-9b`, `flux2-4b` and `flux-2.5` are not accepted for new generation, editing
or variation requests. Adapter choices do not create additional base models.
If `COMFY_URL_VIDEO` is not configured, only Qwen is advertised.

Qwen uses its matching INT8 transformer/text encoder, BF16 VAE and 25 sampling
steps unless a request supplies `steps`. The old global eight-step Flux default
does not override Qwen's model-specific default. LTX retains its distilled 8+3
sampling passes and existing video/adapter contracts.

## Backend placement

`COMFY_URL_IMAGE` configures the primary image backend. It takes precedence over
the legacy `COMFY_URL_FLUX`; unset/blank values fall through to that legacy
variable and then the existing primary-backend default. Docker Compose applies
the same precedence. `COMFY_URL_VIDEO` remains the video backend.

Keep these URLs distinct when backed by different GPUs, allowing Qwen images
and LTX videos to execute independently rather than paying model-switching costs
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
