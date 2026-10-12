# Easel Demo Guides and Source Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Easel agent the same short model guides and connect the shipped video soundtrack input to existing client and CLI flows.

**Architecture:** Keep guide text on the server, expose one read-only lookup, and reuse existing capabilities, transports and tests. Add small explicit audio adapters; avoid a general schema-generation framework.

**Tech Stack:** Existing FastAPI, Python, TypeScript/Zod, MCP SDK, Electron and argparse/httpx.

**Spec:** [Shared contract direction](../specs/2026-10-12-shared-contract-design.md), [reviewed Qwen/H3 guides](../specs/2026-10-12-model-prompt-guides.md), and backend `e4c245cb`. This smaller plan supersedes the previous phase-one plan following the owner's demo-scale correction.

## Global Constraints

- Demo API for a handful of people: no new cache protocol, general code generator, multilayer lock system, PR bot or credentials.
- Preserve authentication, same-origin requests, bounded uploads, saved-asset permissions, accepted receipts and no automatic generation retries.
- Source audio conditions video and supplies its trimmed normalized soundtrack; it is not Suno or selectable upstream copy/reference modes.
- Keep runtime support and successful GPU/visual testing distinct. No deployment, merge or paid generation is included.

## Review Focus

Test missing guides, older deployed capabilities, unavailable saved audio, model-inapplicable fields, and ambiguous accepted submissions in their owning tasks below.

## Current boundaries

LTX timed stills work through saved assets, MCP multipart, and `LTXVAddGuide`/`LTXVCropGuides`. H3 semantic/temporal image groups reach native graph nodes through REST and standalone MCP, but desktop saved-asset adaptation for those H3 arrays is missing. Full reference/control-video upload is unsupported. These are existing limitations, not additions to this task. The new prompt-writing guides currently exist only as documents.

### Task 1: Serve the reviewed guides

**Files:** create `easel/model_guides.py`, `tests/test_model_guides.py`; modify `easel/app.py`, `easel/api_docs.py`.

**Interface:** authenticated `GET /v1/models/{model_id}/guide?mode=...` returns `{version:1, guide_id, model, mode, text, sources}`. Modes are explicit: Qwen generation/editing and H3 base/reference. `get_model_guide(model: str, mode: str) -> dict` reads the single curated registry. Unknown/unreviewed modes return 404; do not borrow H3 syntax for LTX.

- [ ] Write failing tests: correct guide per model/mode, auth required, no upstream generation, 404 for missing guide; run `uv run --group dev pytest -q tests/test_model_guides.py`.
- [ ] Implement the endpoint with existing auth. Use reviewed snippets and source links, including H3 dialogue delimiters. Bound each guide response to 16 KiB. Describe the actual source-audio soundtrack behavior without invented controls.
- [ ] Rerun tests and commit `feat: serve compact model guides`.

### Task 2: Make agents retrieve guides

**Files:** create `packages/media-mcp/src/model-guides.ts`, `packages/media-mcp/test/model-guides.test.ts`; modify `server.ts`, `src/media-mcp-client.js`, `src/agent.js`, `src/easel-tool-host.js`, and `.agents/skills/easel-media/SKILL.md` in easel-client.

**Interface:** read-only `get_model_guide({model, mode})` resolves the existing configured model/credentials, then fetches only its known API origin. Return guide text as reference content, not executable instructions.

- [ ] Test exact model routing, auth reuse, redirects/oversize rejection, unavailable guide, and tool visibility in built-in, embedded Codex and external MCP harnesses.
- [ ] Add the tool and a short shared instruction: fetch the selected model/mode guide before composing a generation prompt; reuse it within the turn. Connect that instruction to each harness and test its presence. Remove conflicting copied model recipes while preserving user-edited imported skills; do not duplicate guide text across consumers.
- [ ] Run package tests and `node --test test/agent.test.js test/easel-tool-host.test.js`; commit `feat: expose shared prompting guides`.

### Task 3: Wire video source audio

**Files:** client `video-input.ts`, `video.ts`, new `source-audio-upload.ts`, `src/media-reference-tools.js`; tests `packages/media-mcp/test/source-audio.test.ts`, `test/media-reference-tools.test.js`. CLI `src/easel_cli/client.py`, `cli.py`, `tests/test_source_audio.py` on existing PR #2 stack `cc5aeaf0` or its merged successor.

**Interfaces:** standalone `sourceAudio:{data,mimeType,name?}`; desktop `sourceAudioAssetId`; CLI `video create --source-audio PATH`. All serialize exactly one multipart `source_audio`. Reuse server `source_audio_capabilities()` and constants; no new policy hierarchy.

- [ ] Write failing tests for byte/asset/file adaptation, null omission, wrong MIME, missing assets, and independent 32 MiB audio/image limits. Require current exact-model `source_audio.available`; absent/unsupported capability blocks only the new argument.
- [ ] Add bounded adapters with existing transport and file-closing patterns. WAV/MP3/FLAC/Ogg/M4A are API inputs; desktop uses already supported saved formats. Server retains decoding, 48 kHz stereo normalization and duration checks: LTX `(seconds*24+1)/24`, H3 `frames/24`. Never add seconds to H3 or silently pad audio. Preserve existing CLI model support rather than adding H3 generation now.
- [ ] Run focused tests plus package/CLI suites; commit each repository's adapter changes separately.

### Task 4: Catch drift with existing tests

**Files:** server `tests/test_mcp_contract.py`, client `.github/workflows/test.yml`, CLI `contract/server_contract.py` and `.github/workflows/ci.yml`.

- [ ] Extend actual consumer→FastAPI/fake-backend tests for source audio, unavailable/old capabilities, short tracks, correct source muxing and one POST after ambiguous failure. Compare shared guide identity and audio field/MIME/limit definitions.
- [ ] Activate these focused tests in existing CI with explicit tested repository revisions and ffmpeg/ffprobe; missing checkouts must fail instead of skip. Use existing access only; report private-checkout blockers without creating secrets.
- [ ] Run relevant full tests/builds, review diffs, then publish coordinated draft PRs with verified heads and results. No live generation is required.

## Tradeoff and handoff

Shared guide prose and server audio policy are DRY; remaining client/CLI schemas stay manual, guarded by focused parity tests. Broader generation, caching and unattended propagation are optional later work.

Self-review: each interface has one owner; all five Review Focus cases have tests; runtime conditions and guide text stay separate; no unrelated H3 desktop work is hidden in scope. Recommend subagent-driven execution with server and consumer owners plus final integrated review. Await plan review and execution-method selection before implementation.
