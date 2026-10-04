# Agent-consumable Easel video controls

## Goal and boundary

The Easel API owns validated graph generation; the Python CLI, standalone Media MCP and in-app/Codex harness consume the same options. Preserve the existing editor and submit-once job IDs. No arbitrary graph JSON, model filenames, server paths or download URLs enter the public generation surface. No merge, deployment or paid generation is part of implementation verification.

Existing server controls are authoritative: camera shorthand, up to four curated LoRA selections, uint64-safe seed, slow-motion conditioning, first-image reference and Ingredients reference sheet. Expose them consistently instead of recreating the Python graph builder in JavaScript. The published creative-skills runtime is LTX-2.3; its structural patterns are evidence, not a claim that unpublished LTX-2.5 code is available.

## HTTP contract

Authenticated GET /v1/videos/capabilities returns object="video.capabilities", schema_version=1, model="ltx-2.5", fps=24, seconds={min:1,max:12,default:4}, sizes (the current seven string IDs), default_size="1280x720", seed={min:"0",max:"18446744073709551614",encoding:"decimal_string"}, loras={max_count:4,min_strength:0,max_strength:2,default_strength:1,camera_default_strength:0.8,catalog_path:"/v1/videos/loras"}, motion_speed={min:0.025,max:1,requires_lora:"slow-motion"}, lora_reference_strength={min:0,max:1,default:1,requires_lora:"ingredients"}, uploads={mime_types:["image/png","image/jpeg","image/webp"],max_total_bytes:33554432}, guiding_frames={supported:true,available:<live required nodes present>,validation:"graph_contract_tested",max_count:8,frame_index_multiple:1,min_strength:0,max_strength:1,default_strength:1,exclusive_with:["input_reference","lora_reference","ingredients"]}. Required guide nodes are LTXVAddGuide and LTXVCropGuides. Gate exact expected input types/output slots, reject unmet required inputs and narrower declared ranges. Guide images must decode as one still image with at most 32 million pixels; reject APNG/animated WebP and malformed/mislabeled/truncated images. Missing node availability is distinct from supported code and GPU/visual validation.

GET /v1/videos/loras stays backward-compatible and preserves supported, installed, requires, validation and provenance. Installation never implies workflow or visual validation. No arbitrary new registry entries are enabled.

POST /v1/videos retains all existing fields. New guiding_frames is a JSON array of strict {image_index,frame_index,strength?} objects; guiding_images is a repeated upload field in matching zero-based order. Metadata must reference each image exactly once. One through eight guides, unique frame indices, integer frame_index in [0, seconds*24], pixel-frame positions (step one), finite strength 0..1 default1. Explicit positions avoid implicit negative-index and rounded-position semantics. Guiding-frame mode is exclusive with input_reference and Ingredients/reference-sheet mode. Camera/regular LoRA compatibility remains registry-controlled; cinemagraph and slow-motion still require input_reference and therefore are unavailable in guiding-frame mode. Reject unknown multipart fields, explicit empty controls, repeated singleton fields, orphan files, malformed metadata, unsupported combinations and oversized total image bytes before queue admission/upload/submission.

## Graph boundary

Retain Easel's native LTX-2.5 UNET/text encoder/VAEs, fixed 24 FPS, 8+3 sampling passes and half-resolution then 2x upscale. Reuse the published graph topology only after lineage/node review. Each guide loads/preprocesses its image, chains LTXVAddGuide on the video latent before AV concatenation in each pass, and passes positive/negative/latent output slots correctly. Crop pass-one guide tokens before spatial upscale; reapply guides to the cropped conditioning after upscale; crop pass-two tokens before decode. Keep original generated audio. No transition LoRA, source-audio, video-control adapters, arbitrary sampler or custom-workflow expansion.

New temporal guides are marked graph-contract-tested until an explicitly opted-in live pilot verifies them. Existing workflow validation labels retain their existing documented evidence.

## Client contract

Media MCP adds discover_video_capabilities({model}) and list_video_loras({model}), using the exact configured model ID to select endpoint and credentials. Discovery is GET-only. generate_video adds cameraLora, cameraLoraStrength, loras (strict typed {id,strength?} array), seed (exact decimal string), motionSpeed, loraReference, loraReferenceStrength and guidingFrames (strict {image,frameIndex,strength?} array). Standalone image is the existing bounded ImageUpload object. In-app schema replaces loraReference with loraReferenceAssetId and guide image with assetId. Every reference uses the same selected project/library resolver and shared 32 MiB combined upload bound. No raw bytes/paths/URLs appear in saved in-app arguments. Strict nested schemas must work with the existing simple harness validator.

CLI adds video capabilities; keeps video loras; typed LoRASelection and GuideFrame inputs; repeated --lora ID[=STRENGTH] and --guide-frame FRAME IMAGE STRENGTH. Keep --loras legacy JSON as mutually exclusive compatibility input, validate before transmission. Exact Python integer seeds remain supported. Explicit regular files are bounded and opened under ExitStack; metadata never contains local paths.

Unknown/older endpoint capability responses fail explicitly. Basic requests keep working without mandatory discovery. Advanced options are never silently dropped. Receipts, resume/get/download, original endpoint credentials and no-automatic-POST-retry semantics stay unchanged.

## Verification

Contract tests cover API inputs through actual graph nodes, CLI multipart through the real FastAPI app with fake ComfyUI, MCP schema/provider/asset bridge and both internal/external routing. Negative cases include unknown keys, malformed/nonfinite values, unsafe seed precision, duplicate and out-of-range positions, mismatched image counts, missing nodes/assets, conflicting modes, invalid references and aggregate byte limit. Assert no upstream work for rejected input, and no resubmission on timeout/resume/failure.

Extend the existing explicit opt-in local-key runner with named bounded advanced cases and read-only capability preflight. Default/help/offline never reads credentials or generates; live requires cost consent, records accepted IDs before polling, resumes via GET and does not replace missing receipts. No authenticated live generation is performed in this task.

## Verified lineage

The current official ComfyUI LTX-2.5 first/last-frame blueprint at e9027f2b30f37bb3052714eb08fcf479542f4fc0 uses Easel’s same int8-convrot UNET and LTXVAddGuide with generated audio. Its single-stage guide chain and CropGuides boundary establish LTX-2.5 compatibility. ComfyUI comfy_extras/nodes_lt.py at that SHA accepts arbitrary nonnegative pixel-frame positions for single-image guides; the published creative-skills LTX-2.3 eight-frame restriction is wrapper policy and is not copied. Applying the guide/crop boundary to both Easel passes is a tested graph composition, not a claim of official two-pass GPU validation. Validate all indices against original seconds*24+1 length because appended guide slots change intermediate latent lengths.
