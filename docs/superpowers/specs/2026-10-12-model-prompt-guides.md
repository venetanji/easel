# Easel model prompt guidance design

Reviewed 2026-10-12. This is authored content for specification review, not production metadata or a live capability claim. Store curated guidance once in Easel and expose it to easel-client, MCP and easel-cli. Select by model and request mode; keep the resulting generation prompt focused on the requested media.

Companion: [shared contract architecture](2026-10-12-shared-contract-design.md).

## Source and applicability policy

- `qwen-image-2.1` identifies native Qwen-Image-2.1. Do not substitute the hosted `qwen-image-2.1-pro` API contract.
- `minimax-h3` identifies native MiniMax-H3. Do not invent an expanded product name.
- Capability discovery controls which operations and uploads are available. Guidance describes how to prompt an available operation; it cannot enable one.
- `official_paraphrase` identifies documented upstream guidance. `easel_recommendation` identifies our authored convention. Illustrations below are original examples, not upstream samples or validated generations.
- The repository snapshot inspected was Easel commit `68d6b2392d0200d6c96ccce5327d7c93060ae55d`. Video source-audio reference changes are in progress elsewhere; these are separate from Suno music/speech/sound generation. Reconcile against the tested backend patch before binding guidance to new request fields.

## Qwen image generation

Guide ID: `qwen-image-2.1.generation`. Applies to text-to-image requests with no input image.

**Suggested snippet, Easel recommendation:** Describe the subject, arrangement, environment, lighting and visual style. Specify visible text exactly. For transparency, explicitly request RGBA, an alpha channel and a transparent background.

**Official basis:** The model card demonstrates descriptive generation, quoted text and an explicit transparency request. It documents one model for generation and editing, including transparent layers. It does not prescribe a universal named-field prompt layout. [Q1]

**Illustration:**

```text
A small blue ceramic teapot centered in a square composition, soft studio light, subtle glaze texture. RGBA output with an alpha channel and a transparent background.
```

## Qwen image editing

Guide ID: `qwen-image-2.1.editing`. Applies to image-conditioned editing. Use reference-order advice only when multiple images are actually supplied.

**Suggested snippet, Easel recommendation:** Name the change and the object or region it affects. State what must stay unchanged. Assign each supplied image a clear role using its input order. For text edits, give exact replacement text. Request transparency explicitly when needed.

**Official basis:** The model card demonstrates a direct background-change instruction and documents identity-preserving edits, local annotations and up to ten references. Our change/target/preserve structure and numbered role convention are recommendations, not a special upstream grammar. [Q1]

**Illustrations:**

```text
Change only the jacket to deep green. Preserve the person's face, pose, hair, background, lighting and crop.
```

```text
Use image 1 as the base scene. Replace its mug with the mug from image 2, matching the existing perspective and light. Keep the hands, table and background unchanged.
```

**Integration cautions:** The inspected graph permits sixteen inputs, whereas Q1 documents ten; do not turn the implementation ceiling into an upstream quality claim. Nor does upstream mask support establish an exposed Easel mask field. Easel rejoins input alpha for conditioning, but generated PNG alpha still needs verification. [E1, E2]

## H3 base video modes

Guide ID: `minimax-h3.base`. Applies to FL8 text-to-video and FL8 first-frame image-to-video in the inspected API. Do not expose separate last-frame or first/last-frame modes merely because the upstream guide covers them. [E3]

**Suggested snippet, official paraphrase:** Use the three fields below in order. Describe composition, subjects, actions and camera movement along the timeline. Keep the image-conditioned opening consistent with the supplied first frame. Separate environmental sound from audience-only music. [H1, H3]

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

**First-frame alignment:** Only when an actual first-frame image is supplied, precede the fields with the official alignment line and a blank line:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

**Essential conventions, official paraphrase:** Camera motion belongs in the shot narrative; include speed and amplitude when useful. Assign stable IDs to speakers and preserve requested speech in language-tagged dialogue blocks. Later shots use increasing cut times inside the clip duration. Use `N/A` for absent score; soundscape `N/A` means complete silence was requested. [H1]

**Short text-to-video illustration:**

```text
integrated_multimodal_description: [Shot 1] A close view of a red paper boat floating beside a stone step. The camera holds still as a small ripple carries the boat toward the right edge.

overall_soundscape: Water laps gently against the step.

non_diegetic_music: N/A
```

**Easel recommendation:** Fit actions and dialogue to the selected duration. For a brief clip, prefer one readable action unless the user needs cuts. Derive duration from the authoritative frame/FPS contract; do not use a rounded display duration to place an endpoint.

## H3 reference video mode

Guide ID: `minimax-h3.reference`. Applies to supported REF20 semantic-reference requests. The inspected API also has experimental temporal groups, which need a separate verified mapping before any automated prompt alignment. Semantic references are not timeline anchors. [E3]

**Suggested snippet, official paraphrase:** Use six fields in the order below. Define reusable subjects and their source images. Keep reference labels consistent. Explain retained and changed attributes, then describe how referenced content participates in each shot. [H2]

```text
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

**Essential conventions, official paraphrase:** A Subject label denotes reusable visual content. Picture labels identify source images or concrete composition/frame anchors; an identity-only source needs no separate Picture definition. Retention distinguishes full preservation, partial preservation, attribute transfer and weak reference. The upstream guide normally recommends a 350–500-word generation description, so a short help snippet must not impose a short final-prompt limit. [H2]

**Abbreviated structural illustration:** This demonstrates field and label mapping only; it is shorter than the upstream production-description recommendation.

```text
subject_definitions:
<Subject 1> is the red paper boat in <Picture 1>, retaining its folded shape and color.

summary:
[reference generation] <Subject 1> drifts across a shallow pool.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - retain its red paper and distinctive folds.

detailed_description:
Natural daylight with a low viewpoint. [Shot 1] <Subject 1> floats past a stone step. A small ripple rotates the boat slowly while the camera remains still.

overall_soundscape:
Gentle water movement against stone.

non_diegetic_music:
N/A
```

**Easel recommendations:** Build reference numbering from the actual request mapping. Do not assume a temporal-guide upload is also a semantic Picture reference. Do not manufacture first/last-frame claims from identity references. Retention language expresses intent, not guaranteed pixel fidelity.

**Audio boundary:** Generated sound guidance above is applicable without an audio upload. This design intentionally defines no new upload fields, copy controls or audio-reference applicability. Add those after inspecting the tested backend schema and reference mapping; do not preserve an obsolete blanket prohibition once that patch lands.

## Metadata for one source of truth

Recommended envelope: `schema_version`, `model_id`, `guide_id`, `guide_revision`, `task`, `mode`, capability applicability, `snippets`, `ordered_fields`, `examples`, `sources`, and `reviewed_at`.

Each snippet should carry its provenance kind and source IDs. Each source should carry a verified URL and revision when known. Keep guide revision separate from model revision and source revision. Reference binding and timing come from the selected request/capability contract, not a second copy of backend limits.

Clients consume the same versioned server payload and cache by server identity, model, mode and guide revision/ETag. No client-specific prose copies, silent prompt rewriting, or unreviewed upstream scraping. Unknown guide schema versions should preserve ordinary prompt entry and show a link or a clear guidance-unavailable state rather than inventing instructions. These are design recommendations, not existing behavior.

## Sources

Q1. [Official Qwen-Image-2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/README.md). Observed file revision `d26bb61231c349cf6b7896fa83353113880e1ba3`, verified through its commit link. Applies to native model guidance, not hosted API restrictions.

H1. [Official MiniMax-H3 base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md). Observed file revision `5d9b308a59ab12e67147f191e184baf704185bd1`, verified through its commit link.

H2. [Official MiniMax-H3 reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md). Observed file revision `5d9b308a59ab12e67147f191e184baf704185bd1`, verified through its commit link.

H3. [Official MiniMax-H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md). Reviewed at the mutable `main` URL; no full commit SHA verified for this file. Confirms mode selection and exact field order. This is source material, not a required installation.

E1. [Easel Qwen graph](https://github.com/venetanji/easel/blob/68d6b2392d0200d6c96ccce5327d7c93060ae55d/easel/qwen_graph.py).

E2. [Easel model serving](https://github.com/venetanji/easel/blob/68d6b2392d0200d6c96ccce5327d7c93060ae55d/MODEL_SERVING.md).

E3. [Easel video API](https://github.com/venetanji/easel/blob/68d6b2392d0200d6c96ccce5327d7c93060ae55d/VIDEO_API.md).

All sources reviewed 2026-10-12. Upstream capabilities and repository support are distinct from deployment availability and visual/audio validation.
