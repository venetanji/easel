# Capacity, authentication and recovery audit

Local audit date: October 9, 2026, Hong Kong time. The main live burst ran
October 8, 2026, 16:06:15-16:09:22 UTC (00:06:15-00:09:22 HKT).

## Readiness verdict

The bounded mixed-media burst passed, including an Easel restart with work
still queued. This is not a large-user soak test, maximum-VRAM benchmark, or
proof of recovery from a killed live GPU worker. Public authentication is fixed
and deployed. The important remaining risks are host RAM exhaustion, low disk
headroom, non-durable ComfyUI queues/history, ambiguous unacknowledged POSTs,
and Suno's single shared browser/confirmation state.

## Authentication and returned URLs

Before the fix, the public Easel hostname returned HTTP 200 for `/v1/models`
both without a token and with a deliberately wrong bearer token.

The path is Caddy on ait4x -> Headscale Serve -> Easel. The backend socket peer
is `10.99.0.8`; captured forwarded headers identify `100.64.0.11` as the caller.
That is the ait4x node, which matched the existing trusted tailnet allowlist.
The Serve hop also replaces Host/forwarded Host with `easel.tail.ait4x.org`.
Consequently neither the outside client's IP nor the public origin reaches
the application unchanged. Merely trusting proxy headers cannot fix this path.

Deployed settings:

```text
EASEL_TRUSTED_PROXY_CIDRS=10.99.0.8/32
EASEL_REQUIRE_AUTH_CIDRS=100.64.0.11/32,fd7a:115c:a1e0::b/128
EASEL_PUBLIC_URL=https://easel.ait4x.org
```

Require-auth CIDRs take precedence over both client and proxy trust, including
every forwarded hop Easel traverses. Uvicorn still uses `--no-proxy-headers`.
The public origin is configured, not taken from a caller-controlled header.

Post-deployment checks:

- Missing token, wrong token, and spoofed tailnet/loopback headers: HTTP 401.
- Correct token: HTTP 200; wrong-token generation POSTs reach none of the backends.
- Wrong-token image/video/audio generation, status and sampled content routes: 401.
- Local access remains 200 without a token.
- A direct Headscale call from `100.64.0.9`, using the Serve hostname, remains 200 even with a wrong token, as intentionally allowed by network authentication.
- Public image and audio URLs now use the public HTTPS origin; fetching the returned URLs with the bearer token succeeds.

Both ait4x addresses were verified in the Headscale peer status; IPv6 is denied
too even though the observed public requests used IPv4.

This exception depends on the public proxy's actual tailnet addresses. Update it
if ait4x changes address or another public ingress node is added. An unset API
key still disables bearer authentication, matching the existing policy.

## Real-generation burst and queue boundaries

The burst sent 26 simultaneous public API requests with a valid bearer token:
12 image requests, 12 video requests and two Suno music requests. GPU queues
were empty beforehand. There were no automatic generation retries.

| Backend | Request | Admitted | Overflow | Verified output |
| --- | --- | --- | --- | --- |
| Image, RTX 3080 Ti 12 GiB | Qwen 2.1, 512x512, 25 steps | 8 x HTTP 202 | 4 x HTTP 429 | 8 PNGs, Pillow verification |
| Video, RTX 3090 24 GiB | LTX 2.5, 512x320, 1 second | 8 x HTTP 200 | 4 x HTTP 429 | 8 MP4s, full FFmpeg decode |
| Suno shared browser | Two 30-second instrumental music requests | 2 x HTTP 202 | None in this two-call burst | 4 complete tracks, playback-audio download and full FFmpeg decode |

Image acceptance was 0.415-1.144 seconds; successful video acceptance was
0.995-1.183 seconds. Suno acceptance was 10.527 and 17.015 seconds: its browser
work serialized even though the two callers connected simultaneously.

Backend execution times, excluding client queue wait:

- Image: median 6.001 seconds, range 5.162-9.703; eight jobs used 51.173 seconds.
- Video: median 20.916 seconds, range 19.549-26.665; eight jobs used 173.855 seconds.
- All 67 sampled health requests returned 200; maximum observed end-to-end latency was 0.955 seconds. The monitor was not running during the intentional restart itself.

The GPU admission limit is eight total running plus pending prompts per backend,
not eight simultaneous executions. Each backend executes one prompt at a time;
the two separate GPUs can execute image and video work in parallel. These are
small workloads, not a claim about default-size videos, H3, long clips, image
batches, or maximum-resolution memory safety. A prompt can fan out into multiple
images, so eight prompts need not mean eight images.

A separate 20-request burst against a real generated Suno track's metadata
returned nine HTTP 200 responses and eleven HTTP 429 responses. This exercises
the same browser admission used by generation without spending more credits:
one active operation plus eight waiting operations. Queue wait expires after
15 seconds; operations have a 120-second deadline. Long browser operations can
therefore produce 503 queue timeouts before they execute. Track metadata reads
and download preparation compete with generation for this same browser;
streaming already-saved audio does not need the browser lock.

## Live restart and disconnect results

After all acceptance responses were saved, the submitting clients closed their
connections. Easel was restarted with one image running plus one pending, and
one video running plus six pending. The same saved IDs were polled after restart;
all eight images and eight videos completed without replacement submissions.
The four Suno track IDs were still complete and downloadable.

Two additional real image probes closed their TCP connection before reading any
response, while the prompt was running:

- Synchronous image: disconnected after 0.029 seconds; ComfyUI still completed it, but Easel had no durable receipt/cache row for that prompt.
- Async image: disconnected after 0.032 seconds; ComfyUI still completed it, and Easel retained a completed receipt/cache row. The disconnected caller nevertheless did not know that ID.

An earlier observer-only probe also submitted one small image. Its queue
observer incorrectly searched a `text` field rather than Qwen's `prompt` field;
the accepted prompt was subsequently found completed in history, without
resubmitting it. It is excluded from the burst's admission/timing statistics.

## What survives, and what does not

| Failure | Result | Recovery |
| --- | --- | --- |
| Client disconnect after saving async receipt | Image/video generation continues | Poll the same ID, not a new POST. |
| Client disconnect before receipt arrives | Submission is ambiguous | No idempotency key or client job-list API exists. An operator may locate an image in SQLite/history, but the caller has no guaranteed recovery handle. |
| Easel process/container restart, Comfy stays up | Accepted async images and encoded video IDs survive | Keep the image-jobs volume and client receipts; poll those IDs. Sync HTTP responses are lost. Video ETA samples reset. |
| Comfy process death/restart | Running work, pending queue and API history are in memory and disappear | Unfinished async images become `failed/upstream_job_lost` after about 60 seconds of successful observations finding no job; videos become 404. No automatic replay occurs. |
| Comfy unavailable but its process/state survives | Jobs are not declared lost merely because HTTP is unavailable | Restore connectivity and poll the existing receipt. |
| Comfy restart after an image has been cached | Easel's completed image bytes survive | Retrieve from the persistent Easel cache, within the 24-hour TTL. Completion not yet cached is still vulnerable. |
| Comfy restart after video completion | Output files can remain on disk, but API lookup/history is lost | Download completed videos promptly. There is no automatic file-to-video-ID reconstruction/cache. |
| Suno/Easel disconnect or timeout during browser submission | Accepted remote work may continue; confirmation can be ambiguous | Never blindly repeat Create. Retain attempt/track IDs and check status/library. |
| Suno backend restart | Waiting browser operations are lost; the latest generation state and Chrome profile are persisted | An unresolved restored attempt is guarded with `recovery_required`, requiring browser/library inspection. This is not a durable multi-user generation queue. |
| Remove persistent volumes/directories | Cached images, local audio and/or saved browser state are lost | Restore backups; remote Suno tracks may still exist. |

An isolated subprocess test used the installed ComfyUI `PromptQueue` class,
populated one running prompt, one pending prompt and one history entry, then
SIGKILLed that subprocess. A fresh process had zero of all three. The actual
GPU services were not killed or restarted for this experiment.

Regression tests also cover missing backend jobs, completed image-cache recovery,
video lookup loss on backend restart, and backend acceptance followed by a lost
reply/cancellation. In the latter window, the accepted backend prompt can exist
without an Easel receipt: the reservation is discarded and no replay is attempted.

The Suno persistence restore method was tested with a temporary directory and
an unresolved saved attempt, inside the running container but without touching
the real profile/browser. It restored the same attempt ID with
`recovery_required:true` and `recovery_reason:backend_restarted`. A live Suno
process crash and CAPTCHA interaction were not forced during this audit.

## Historical ComfyUI crash: confirmed host OOM

On September 28, 2026, at 21:54:17 HKT, the kernel killed Python PID 1281185 in
Docker container `93d5aa788252...`. Its anonymous RSS was 19,090,244 KiB
(approximately 18.2 GiB). The kernel explicitly reported `global_oom` and zero
free swap; total swap was 3,773,436 KiB. Other browser/client processes were also
killed during this host-wide memory exhaustion.

The retained September 29 shell output maps that container ID to
`comfy-docker-comfyui-1`, the primary image ComfyUI service. Docker recorded exit
code 137 and an automatic `unless-stopped` restart at 21:54:19 HKT. This is a
confirmed Comfy process death from host RAM/swap exhaustion, not a recorded
CUDA-VRAM OOM. The available records do not prove which workload caused the
aggregate host memory exhaustion.

Current GPU containers started on October 4 and show no automatic restart/OOM
kill. Their retained histories contain 11 handled workflow failures on October
6 UTC, not process crashes: nine `LoadImage` nodes with an empty filename, and
two Flux 4B graphs using an incompatible Qwen 8B text encoder, producing matrix
shape mismatches. Current Qwen/LTX audit generations all completed successfully.

The kernel/Docker excerpts and historical container-name mapping are saved in
`audits/2026-09-28-comfy-oom.txt`.

## Before a large-user trial

1. Address host RAM pressure first. At audit start the host had about 21 GiB available, but its 3.6 GiB swap was already full. The previous global OOM actually killed ComfyUI. Coordinate other memory-heavy services, monitor RAM/swap during the trial, and set explicit memory/resource policies rather than increasing concurrency blindly.
2. Free disk space and establish output retention/monitoring. The shared root filesystem was 95% used, with about 52 GiB available. Images, videos, audio, SQLite and upload spooling all share this failure domain.
3. Use async image receipts and persist receipts on clients. Default synchronous image admission currently allows only one in-flight request per backend; increasing connection count does not increase GPU execution throughput. Handle 429/503 with bounded backoff, but do not retry an ambiguous accepted generation.
4. For reliable recovery, add a durable request/graph ledger, idempotent admission and durable video output/metadata caching. Current recovery cannot guarantee reconstruction after a backend process crash or a lost acceptance reply. Paid Suno submissions must retain explicit no-blind-retry semantics.
5. Treat Suno as one shared account/browser with one latest global confirmation state, not independent per-user sessions. Add per-user ownership, fairness and quotas before promising isolated multi-user jobs. Avoid frequent metadata polling or download preparation that competes with Create.
6. Keep one Easel worker with the current process-local admission locks. Multiple workers/instances can race video queue checks. Restrict direct ComfyUI ports: the current host publishes 8189/8190 on all interfaces, and direct ComfyUI access bypasses Easel's bearer authentication and eight-prompt admission cap. WAN reachability was not tested. Add upload/body limits as part of ingress hardening.

## Reproduction and artifacts

Use `scripts/audit_capacity.py --live` only in an agreed idle window: it submits
real GPU jobs and paid Suno requests. Its default burst matches this audit.
Reports are written atomically with owner-only permissions; uncertainty is saved
before POSTs and receipts before polling. `--submit-only` allows an intentional
Easel restart; `--resume` polls saved IDs without submitting new generations.

Raw evidence in `audits/` includes the main burst/verified media metadata,
Suno browser queue burst, TCP disconnect probes, isolated Comfy queue death,
post-deployment authentication/public-URL checks, and historical OOM excerpts.
Local decoded media is in `/tmp/easel-capacity-media-2026-10-09/`.

The standard suite remains fake-backend-only unless explicitly opted into live
integration. This audit separately used real backends; no live GPU worker was
deliberately killed and no arbitrary CAPTCHA was bypassed.

Final standard validation: 785 tests passed, eight opt-in integrations skipped,
and two existing dependency deprecation warnings. `git diff --check` passed.
Both GPU queues were empty at handoff and the Easel service was healthy.
