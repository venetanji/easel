# Video sizing, admission and GPU verification

Agents should discover `GET /v1/videos/capabilities?model=ltx-2.5` or
`?model=minimax-h3` instead of maintaining their own resolution whitelist.
Existing `sizes` and `default_size` fields remain available. Additive fields:

- `size_presets`: named sizes, actual width/height and `max_frames`; LTX also
  reports the maximum integer `seconds` as `max_seconds`.
- `size_constraints`: custom-size format, grid, bounds, aliases and the deployed
  pixel/pixel-frame budgets. `oom_guarantee:false` is intentional.
- `failure_recovery`: OOM error code and an explicit no-automatic-retry policy.

`size_presets[].validation` is `admission_only`: passing a budget check is not
GPU validation. The common presets include square, landscape and portrait
outputs. A model only advertises presets that fit at least its minimum duration;
H3 additionally retains its original `864x480` preset. Custom aligned dimensions
are accepted under the same limits, even when not in `sizes`.

## Alignment and duration

LTX samples at half size on a 32-pixel grid, then upscales 2x: final dimensions
must be multiples of **64**. H3's native recipe uses multiples of **32**.
Dimensions must be in `256..2048`. Invalid custom sizes are rejected, not rounded.
`1280x720` and `720x1280` remain compatibility aliases for the actual aligned
`1280x704` and `704x1280` outputs on both models.

LTX accepts integer seconds in `1..12`, using `seconds * 24 + 1` frames. H3 uses
its existing `17k+5` frame grid in `124..362`, at 24 FPS. New IDs preserve actual
size and timing across Easel restarts; old IDs remain readable.

## Resource limits

Both models obey two deployment settings:

| Setting | Default | Meaning |
|---|---:|---|
| `EASEL_VIDEO_MAX_PIXELS` | 2,097,152 | Maximum `width * height` |
| `EASEL_VIDEO_MAX_PIXEL_FRAMES` | 160,000,000 | Maximum `width * height * output_frames` |

Admission enforces the effective aligned output size, including aliases.
Over-budget requests fail with `400`, code `video_resource_limit`, the relevant
`size`/`seconds`/`frames` parameter and limit details, before queue discovery,
runtime discovery, uploads or submission. The budget is inclusive; the largest
allowed duration is rounded down to the model's temporal grid.

With default settings, these are **admission ceilings, not a GPU-tested matrix**:

| Actual size | LTX max seconds | H3 max frames |
|---|---:|---:|
| `512x512` | 12 | 362 |
| `640x640` | 12 | 362 |
| `768x768` | 11 | 260 |
| `1024x1024` | 6 | 141 |
| `1280x1280` | 4 | Not admitted at H3's minimum length |
| `1280x704` | 7 | 175 |
| `1792x1024` | 3 | Not admitted at H3's minimum length |
| `1920x1088` | 3 | Not admitted at H3's minimum length |

Memory is not linear in output pixel-frames. Attention, reference images,
temporal guides, adapters, decoding, precision, offloading and other GPU users
affect peaks. These are portable workload guardrails, not a VRAM calculator or
a promise that every admitted job fits. Benchmark the actual deployed recipe,
model revision and GPU before increasing limits. If an operator reduces limits
below the normal default request, callers must explicitly choose a smaller size
and/or duration; the historical defaults are not silently changed.

Raw ComfyUI/comfy-graph submissions bypass Easel's admission policy. Agents using
that route must apply the same advertised constraints themselves, or treat their
submission as a separate explicitly authorized experiment.

## OOM and recovery

A handled backend memory failure returns a terminal failed video:

```json
{
  "code": "upstream_out_of_memory",
  "message": "ComfyUI ran out of memory; reduce size or duration before retrying",
  "retryable": false,
  "suggested_action": "reduce_size_or_duration"
}
```

`retryable:false` means do not blindly repeat the same workload. A caller may
explicitly submit a smaller workload after checking backend health/queue state.
Easel does not retry, interrupt, call a global memory flush from a read-only
poll, or restart ComfyUI. Standard ComfyUI catches its handled OOM exception and
unloads loaded models before recording the execution error; this cleanup was
also confirmed in the deployed native `execution.py`. A failed job does not
hold an Easel submission lock. Subsequent jobs can use the same queue.

An OS OOM kill, dead driver or crashed backend is different: it may leave no
terminal history. Treat network errors/missing history as uncertain outcomes,
preserve the receipt and reconcile the backend before resubmitting. Never
restart a backend with another owner's active generation merely to recover a
failed benchmark.

## Opt-in live matrix

`scripts/benchmark_video.py` is dry-run by default. It reads live capabilities,
validates **all** cases before GPU submission, and runs at most eight jobs
serially. Supply authentication through `EASEL_API_KEY`, not command arguments.
Keep media, reports and credentials outside Git.

```bash
uv run python scripts/benchmark_video.py \
  --model ltx-2.5 \
  --case 512x512:1 --case 768x768:2 --case 1792x1024:1 \
  --output /tmp/easel-video-pilot
```

Cases are `SIZE:SECONDS` for LTX and `SIZE:FRAMES` for H3. Confirm the target
ComfyUI queue is idle and add `--execute` to submit. Optional `--gpu-index 1`
samples local total-device usage via `nvidia-smi`; this is a sampled peak, not
isolated per-job allocation, and can include other processes/cached models.

The runner saves `report.json` privately and atomically before each submission,
then persists the receipt before polling. It downloads completed output and
checks actual width, height, decoded frame count, 24 FPS, an audio stream, and
a full `ffmpeg` decode. It stops on the first failure/OOM/timeout/verification
error, without submitting the next case or automatically retrying.

Resume an accepted job after a timeout or lost polling connection:

```bash
uv run python scripts/benchmark_video.py \
  --output /tmp/easel-video-pilot --resume --execute
```

Existing receipts are polled, not replaced. If a submission lost its receipt,
the report stays in `submitting` and resume refuses a replacement; reconcile
the backend manually. Retain evidence per resolution/duration **and conditioning
profile**, and do not extrapolate a short unconditioned pilot to a maximum-length
reference/adapter workflow.

## Verified deployment pilot: October 6, 2026

Unconditioned LTX-2.5 T2V runs on the local RTX 3090 (24 GiB, NVIDIA driver
615.71.09), with native ComfyUI revision
`b5cc8830279eae909a59de030af1e50761c36751`, passed these requested combinations:

| Actual size | LTX integer seconds verified |
|---|---|
| `512x512` | 1, 12 |
| `768x768` | 2, 11 |
| `1280x704` | 4, 7 |
| `1280x1280` | 4 |
| `1792x1024` | 1 |
| `1920x1088` | 3 |

The same deployment also passed native MiniMax H3 **FL8 T2V** combinations:

| Actual size | H3 frames verified | Video duration |
|---|---:|---:|
| `864x480` | 124 | 5.167 seconds |
| `640x640` | 124 | 5.167 seconds |
| `1024x1024` | 141 | 5.875 seconds |
| `512x512` | 362 | 15.083 seconds |

All thirteen outputs match their requested dimensions, expected frame count and
24 FPS, contain audio, and pass full decoding. No OOM occurred. Device-wide
sampled peaks reached 24,117 MiB out of 24,576 MiB, including cached/resident
models; there is limited observed headroom and this is not a free-VRAM estimate.
Artifacts and detailed private reports remain outside Git. These pilots do not
validate other GPUs, every admitted geometry, I2V, additional adapters, semantic references,
temporal guides, or visual/audio quality. The discovery labels intentionally
remain `admission_only`, rather than promoting all durations/profiles as tested.
