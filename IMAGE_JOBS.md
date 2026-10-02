# Async Image Jobs

This implements the proposed contract in Easel issue #1 and easel-client PR #1
(client adapter inspected at `91427cb31af09bdf1e459ed08097b1f546b7a7b3`).
Existing clients without the preference continue receiving synchronous HTTP 200
image responses.

## Submit once

Send `Prefer: respond-async` to any of these existing endpoints:

- `POST /v1/images/generations` (JSON)
- `POST /v1/images/edits` (multipart)
- `POST /v1/images/variations` (multipart)

All existing model, size, prompt, reference, response-format and `server` fields
retain their meaning. Async admission happens before reference uploads. The
selected image/video backend is stored with the receipt; subsequent polling
does not need to repeat the `server` parameter.

A successful ComfyUI acceptance returns HTTP 202, `Preference-Applied:
respond-async`, `Location: /v1/images/jobs/{id}`, and `Retry-After: 2`:

```json
{
  "id": "image_job_1790000000_0123456789abcdef0123456789abcdef",
  "object": "image_job",
  "model": "qwen-image-2.1",
  "status": "queued",
  "created_at": 1790000000,
  "expires_at": 1790086400,
  "completed_at": null,
  "progress": null,
  "error": null,
  "queue_position": 1,
  "queue_ahead": 0,
  "running": 0,
  "pending": 1,
  "average_generation_seconds": null,
  "estimated_wait_seconds": null,
  "estimated_completion_at": null
}
```

The accepted ID is persisted before the receipt is returned. Save it immediately.
Poll `GET /v1/images/jobs/{id}` with the original bearer credentials. Polling is
read-only with respect to generation: it never submits another ComfyUI prompt.

## Completion and content

Completed responses retain the receipt fields and include normal image output:

```json
{
  "id": "image_job_1790000000_0123456789abcdef0123456789abcdef",
  "status": "completed",
  "progress": 100,
  "estimated_wait_seconds": 0,
  "data": [{"b64_json": "..."}]
}
```

With the original `response_format=url`, `data` instead contains URLs on the
same Easel origin, pointing to `GET /v1/images/jobs/{id}/content/{index}`.
Indices are zero-based. These URLs require the same bearer credentials; they
are not public or signed third-party download links. Content is served from
Easel's cache, not from arbitrary caller-supplied paths or URLs.

A lifespan worker observes accepted jobs and caches their outputs even when
the client is disconnected. The cache becomes terminal atomically, only after
successful ComfyUI history and retrieval of all intended images. Partial preview
outputs do not count as completion. Cache limits are 16 images / 128 MiB per job.

## Lifecycle, estimates and limits

- States: `queued`, `in_progress`, `completed`, `failed`, `cancelled`.
- Progress is `null` until completion, then `100`; Easel does not invent measured
  sampler progress. Failures include a compact `error.code` / `error.message`.
- `queue_position=0` means running; pending positions start at 1.
  `queue_ahead` counts backend jobs ahead, including external ComfyUI jobs.
- `estimated_wait_seconds` means **time until this job completes**, not just
  queue waiting. It includes remaining active work and this job's own generation.
  `estimated_completion_at` is Unix seconds. Estimates can change with queue
  movement and elapsed runtime; they are not promises or quality-specific models.
- Estimates are `null` until a successful execution supplies timing data. The
  rolling mean uses up to 20 successful cached jobs on that backend. Different
  models, sizes, batch counts and external jobs can make the mean inaccurate.
- Queue counts are `null` when no backend snapshot is available, including
  cached terminal retrieval without contacting ComfyUI.
- HTTP 429 + `Retry-After: 5` means the async reservation or backend queue is
  full. There are at most 8 outstanding async image reservations per backend.
  This is separate from `COMFY_MAX_INFLIGHT`, which still limits synchronous
  requests. Admission is atomic across workers sharing the SQLite directory.

## Restart and retention

`EASEL_JOB_DIR` holds the SQLite receipt/output cache. For a bare process, its
default is `~/.local/share/easel/jobs`; Docker Compose sets `/app/jobs` on the
persistent `image-jobs` volume. Keep that directory/volume across redeployments.

Accepted IDs survive Easel restarts. Completed, cached output also survives a
ComfyUI restart or history clear. An unfinished ComfyUI job cannot be resumed
after ComfyUI discards it: after 60 seconds absent from queue/history it becomes
`failed` with `upstream_job_lost`, never a replacement submission. Backend HTTP
outages return an upstream error/defer observation; they do not immediately mark
the job lost. A crash during an unacknowledged submission can leave a reservation,
which is eventually failed without resubmission.

Receipts and cached output expire 24 hours after creation. Retrieval of an expired
well-formed ID returns HTTP 410 (`image_job_expired`); malformed/unknown IDs return
HTTP 404 (`image_job_not_found`). The worker removes expired rows/output from its
own cache. SQLite reuses freed pages; its file may retain its high-water size.
Removing the volume loses image receipts and cached output.

**Idempotency is not implemented.** A client crash or network failure before it
receives the receipt remains an ambiguous submission. Do not automatically retry
the generation POST. An `Idempotency-Key` is not currently a recovery mechanism.

Forgetting a generating card only stops client monitoring; it does not cancel
server execution. There is no cancellation endpoint in this increment. The
`cancelled` state reports a backend interruption. Status/content authentication
matches the original submission.

Video IDs still use the separate ComfyUI-backed contract in `VIDEO_API.md`;
video output has not been migrated into this image cache.
