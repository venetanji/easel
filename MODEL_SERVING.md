# Model Serving and Queue Admission

This server revision advertises image models `qwen-image-2.1`,
`moss-image-base` and `moss-image-sft-8step`, plus `ltx-2.5` and `minimax-h3`
when `COMFY_URL_VIDEO` is configured. Image requests with an omitted/blank model
use Qwen; video requests require an explicit `ltx-2.5` or `minimax-h3` model.
This describes code support, not independently verified deployment state.
`flux2-9b`, `flux2-4b` and `flux-2.5` are not accepted for new generation, editing
or variation requests. Adapter choices do not create additional base models.
Without `COMFY_URL_VIDEO`, neither video model is advertised.
Setting `SUNO_URL` independently adds `suno-music`, `suno-speech`, and `suno-sound`.
These audio routing IDs use the separate Suno REST/browser backend, not ComfyUI;
see [AUDIO_API.md](AUDIO_API.md) for options, status polling and saved audio.

Qwen uses its matching INT8 transformer/text encoder, BF16 VAE and 25 sampling
steps unless a request supplies `steps`. The old global eight-step Flux default
does not override Qwen's model-specific default. LTX retains its distilled 8+3
sampling passes and existing video/adapter contracts.

## MOSS-Image Base

Use `model: "moss-image-base"` on `/v1/images/generations`, preferably with
`Prefer: respond-async`. This backend reuses the installed OpenMOSS HF-directory
plugin, not the newer native SFT loaders. ComfyUI must expose `MOSSImageHFLoader`,
`CLIPTextEncode`, `EmptyFlux2LatentImage`, `ModelSamplingSD3`,
`SkipLayerGuidanceDiTSimple`, `KSampler`, `VAEDecode`, and `SaveImage`.
Prompt fan-out also requires `SimplePromptBatcher`.

The model directory is `/app/ComfyUI/models/moss-image/MOSS-Image-Base` inside
ComfyUI, containing `transformer/`, `text_encoder/`, `tokenizer/`, and `vae/`.
The registry owns this path; clients cannot supply arbitrary model paths.
Weights are not downloaded automatically. The existing directory occupies
about 26 GiB. FP8 diffusion loading and CPU text encoding fit the image GPU;
the text encoder alone needs about 16 GiB of host RAM. Start with batch size 1.
The recipe uses CFG 3, shift 5, Euler / sgm_uniform, Simple SLG layer 22, and
50 steps by default. The common size, seed, steps, n, separator, server and
response-format controls apply; default size is 1024x1024.

Edits and variations return `400 unsupported_operation` before uploads,
admission or submission. Native layout prompting and edit LoRAs are not
integrated. SFT is available through the distilled profile described next.
Read `/v1/models/moss-image-base/guide?mode=generation` for guidance.
The 31 October direct ComfyUI smoke completed at 512x512, 20 steps and seed 42
in 34.851 seconds. After deployment, an authenticated public async request
with omitted size/steps completed a fresh 1024x1024, 50-step, seed-43 generation
in 57.76 seconds of ComfyUI execution and served a valid 519346-byte PNG.
These runs do not establish every size/batch or pixel-art/transparency quality.
See [GRAPH_PROVENANCE.md](GRAPH_PROVENANCE.md).

## MOSS SFT Eight-Step DMD

Use `model: "moss-image-sft-8step"` on `/v1/images/generations`. This curated
profile applies the user's selected
[lodestones lora.safetensors](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/blob/f95df83b5e90ae91ea412434b3f9b7f1c8c6314f/lora.safetensors)
at strength **1.0** to the matching **SFT** transformer. It uses 8 steps by
default, CFG **1**, shift 5, Euler / sgm_uniform and **no SLG**. Diffusion weights
are loaded as BF16 so small adapter changes are not rounded into FP8 weights;
text encoding remains on CPU. The normal image size, steps, seed, batch,
separator, backend and durable async receipt controls apply. Default size is
1024x1024. LoRA choice/strength and guidance are fixed by the profile.

ComfyUI additionally needs `LoraLoaderModelOnly` and these local assets:

- `/app/ComfyUI/models/moss-image/MOSS-Image-SFT/transformer/`: the three official
  SFT BF16 shards, config and index at revision
  `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`.
- The SFT directory's `text_encoder`, `tokenizer` and `vae` are relative symlinks
  to `../MOSS-Image-Base/` components. Their repository blobs match SFT, so no
  second copy is needed. Keep that shared Base directory when using this profile.
- `models/loras/moss_image_sft_8steps_dmd.safetensors`: `lora.safetensors` from
  `lodestones/MOSS-Image-SFT-8-steps-dmd@f95df83b5e90ae91ea412434b3f9b7f1c8c6314f`,
  SHA-256 `4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe`.

Additional weight storage is about 10.6 GiB. Assets are not downloaded by Easel;
the installed SFT shards and adapter were verified against their repository
SHA-256 values. The similarly named `lora_latest.safetensors` recommends a
different strength and is not used here. The model card's deprecated custom
data-space loop and extra generator/time-embedding files are not used.

A 1024x1024, seed-43 test applied all 252 rank-128 adapter modules and completed
eight-step generation in 30.403 seconds including cold loading and CPU text
encoding. A same-prompt SFT BF16 50-step/CFG-3 control completed in 57.158 seconds;
an eight-step/CFG-1 control without the LoRA completed in 5.310 seconds but had
softer, less defined character details. These are illustrative single-seed
observations; the author labels the adapter work in progress. Cold and warm
execution times are different and should not be treated as a direct speedup.

After deployment, a fresh seed-44 public authenticated API request omitted
size and steps and ran at 1024x1024/eight steps. ComfyUI execution took **6.679
seconds** with model/text caches warm, and polling plus authenticated PNG
retrieval took about nine seconds after acceptance. KSampler was not cached.
This warm observation is about 8.6 times faster than the single warm 50-step
SFT control; it is not a throughput estimate or a quality-equivalence claim.

Only generation is integrated. Edits and variations reject before uploads;
native layout controls, edit adapters and universal quality equivalence are not
established. Read `/v1/models/moss-image-sft-8step/guide?mode=generation`.

## Qwen Editing And Video

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
