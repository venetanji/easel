# LTX Adapter Inventory and Validation

`easel/video_loras.json` is the pinned curated inventory for the video API and
download helper. Advanced model/workflow support belongs in canonical Comfy Graph;
Easel's existing local graph extensions are an interim consumer implementation,
not a completed migration of workflow knowledge into the shared package.

## What is available

The catalog covers 38 entries across Lightricks' LTX-2, LTX-2.3 Creative Lab and
LTX-2.5 Creative Lab collections. On the verified Thor deployment, all 12
LTX-2.5 Creative Lab weights are installed and checksum-verified. This does not
imply that all 38 catalog entries are installed or runnable.

| Adapter | Easel path | Validation boundary |
|---|---|---|
| Seven camera controls | `camera_lora` or registered `loras` IDs | Shape audit and real T2V execution tests passed; visual direction/fidelity still needs review |
| Cinemagraph | I2V, `loras`, trigger added automatically | Real I2V execution passed at 512x320 / 2 seconds; effect fidelity remains experimental |
| Slow Motion Control | I2V, `loras`, `motion_speed` | Real I2V execution passed at 512x320 / 2 seconds / speed 0.2; effect fidelity remains experimental |
| Ingredients | `loras`, `lora_reference`, 5-12 seconds | Real IC execution passed at 640x384 / 5 seconds; reference fidelity remains experimental |
| Clean Plate, Colorization, Day to Night, Deblur, Decompression | Discoverable but rejected | Need their dedicated source-video IC workflows |
| Pixel Upscaler, Refine Details, Restore, Water Simulation | Discoverable but rejected | Need specialized V2V/upscale/simulation workflows and validation |
| Other collection versions/control adapters | Discoverable but rejected | Model/profile/workflow compatibility must be validated separately |

The camera adapters retain their original `ltx-2-19b-...` filenames. They are not
rejected based on model-name labels: safetensors target shapes match the deployed
LTX-2.5 INT8 ConvRot transformer. Each dolly matches 480 targets; jib/static each
match 1,248; none have missing/mismatched target layers. All seven rendered in
the two-pass workflow at strength 0.8, 2 seconds, fixed seed 424242, 512x320,
49 frames. The baseline and all seven MP4s passed full ffmpeg decode.

This is technical execution evidence, not proof that each clip obeys its camera
direction, that I2V/adapter stacks have been live tested, or that every resolution
fits the 24 GB GPU. Camera/art-direction evaluation should happen with the
Creative Agent where the videos can be reviewed in Telegram.

Cinemagraph and Slow Motion each completed with 49 frames at 24 FPS; Ingredients
completed with 121 frames at 24 FPS. All three passed full audio/video decode.
The executed Ingredients graph contains both IC guide/crop chains; Slow Motion
uses conditioning FPS 120 while output stays at 24. Representative first/middle/
last frames were inspected: they are valid scene imagery, not blank/black output.
The Ingredients test reinterprets scene framing/geometry, and the short Slow
Motion/Cinemagraph tests do not establish quantitative effect fidelity. A contact
sheet and native videos are included in the Creative Agent's handoff evidence.

## Verified downloads

The LTX-2.5 collection includes Cinemagraph, Slow Motion, Clean Plate, Colorization,
Day to Night, Deblur, Decompression, Ingredients, Pixel Spatial Upscaler, Refine
Details, Restore and Water Simulation. Total weights are about 9.1 GiB.

Install only requested catalog entries into an explicit ComfyUI LoRA directory:

```sh
python scripts/download_video_loras.py \
  --destination /path/to/ComfyUI/models/loras \
  --token-env-file /path/to/private/backend.env \
  --id refine-details --id ingredients
```

With no IDs/collection option the helper selects `ltx-25-creative-lab`. It checks
existing files, refuses to overwrite mismatched weights, downloads atomically,
checks size/SHA256 and reports access failures. Auth is removed on cross-host
redirects. Keep tokens/private environment files, weights and generated videos
out of Git. Additional older collections are opt-in, not silently downloaded.

## Refine Details and the new tiled-fusion workflow

Refine Details is a video-conditioned refinement/upscale model, not a plain LoRA
patch on T2V. Its guide is the source clip resized to the output canvas, aligned
1:1 in frames/FPS, at reference downscale factor 1. The official
`example_workflows/2.5/LTX-2.5_V2V_TiledFusion_Upscale.json` uses IC guide handling,
per-step tiled latent fusion and source-audio preservation. Canvas dimensions
must be multiples of 32 and frame count `8n+1`. Prompt for look/grade/detail rather
than naming objects that could repeat in each tile. Generative refinement can
change texture, faces and text; it is not lossless reconstruction.

The installed old `LTXVTiledSampler` is not the new `LTXVTiledFusionSampler`.
The latter comes from **Lightricks/ComfyUI-LTXVideo**, not a separate weight.
At inspection, the Docker-volume checkout was `dfb2786` (September 17, 2026);
latest upstream master was `bf2ca0264f706db64cb8931155695ca481fc9d91`
(September 30, 2026), which registers the fusion sampler. The new workflow also
uses newer streaming-guide and source-audio reference-token functionality.

For Thor, update the actual checkout inside
`comfy-docker-comfyui-video-1:/app/ComfyUI/custom_nodes/ComfyUI-LTXVideo` on the
`comfy-docker_custom_nodes` volume, not an unrelated host copy. Safe procedure:

1. Check current queue, revision, local diff and requirements; wait for active
   shared work to finish before changing/restarting the backend.
2. Back up tracked local modifications. The existing `pyramid_blending.py` patch
   uses `torch.nn.functional.pad` because installed Kornia removed its old export.
   Preserve/reapply that behavior if upstream does not already fix it.
3. Fetch and fast-forward to a reviewed immutable upstream revision; resolve
   conflicts explicitly. Do not use a destructive reset or delete backups.
4. Install/check requirements in ComfyUI's actual Python environment, validate
   imports, then restart only the intended video backend during an idle window.
5. Confirm `/object_info/LTXVTiledFusionSampler`, `LTXVGetTilingSizes` and the
   updated streaming-guide schema used by the workflow,
   then run a bounded V2V smoke before advertising high-resolution support.

ComfyUI was not updated/restarted as part of the download/handoff. Existing
models suffice for investigation; obtaining the node does not require downloading
the full BF16 transformer/text encoder. Old callers must be regression-tested.

## Creative Agent handoff

On Thor's Gateway workspace, the durable note is
`memory/2026-09-30-easel-lora-handoff.md`, with camera preview MP4s, pinned catalog,
shape audit, download report and unchanged official workflow under
`media/easel-loras-2026-09-30/`. That workspace is host-mounted at
`/home/venetanji/.openclaw-gateway/workspace-creative-skills`, appearing inside the
Gateway as `/home/venetanji/.openclaw/workspace-creative-skills`.

At the download audit, root storage was 87% used with about 120 GiB free. No old
weights, Docker images/volumes or user caches were pruned. Avoid duplicate
downloads and inspect free space before extending the inventory.
