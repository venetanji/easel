# Easel Shared Contract Phase One Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate the shipped video source-audio contract to Media MCP, the desktop asset interface, and the existing CLI stack, with server-owned metadata/guides and required cross-repository drift checks.

**Architecture:** Extract the source-audio policy already enforced by Easel into one validated definition and deterministic export. Generate its consumer types, upload descriptors and field mappings, while retaining existing transports, video fields and explicit binary/asset adapters. Runtime metadata supplies supported capabilities and compact guides at stable turn boundaries; it never executes downloaded code or enables unknown operations.

**Tech Stack:** Python/FastAPI/Pydantic, ffmpeg/ffprobe already required by Easel, TypeScript/Zod and the existing MCP SDK, Electron/CommonJS, Python argparse/httpx, existing pytest and Node test runners. No new runtime dependency is required for this slice.

**Spec:** [Shared contract design](../specs/2026-10-12-shared-contract-design.md) and [model prompt guides](../specs/2026-10-12-model-prompt-guides.md), merged in PR #8. Backend baseline: [e4c245cb](https://github.com/venetanji/easel/commit/e4c245cb17a9dd8bf9d49b872e87d6b90ecd9390).

## Global Constraints

- Design review and implementation planning only until this plan and execution method are approved.
- “Keep the canonical contract in `venetanji/easel`, beside the implementation.”
- “No manual edits to generated output.”
- “Pin a coherent metadata/schema snapshot for each active agent turn.”
- “A metadata failure never triggers generation, a settings change, or automatic POST replay.”
- “New upload types, reference indexing, request content types, authentication behavior, operations, or lifecycle semantics always require a reviewed adapter change and tests.”
- “No GPU generation, Suno charges, deployment or live-secret testing is required for this CI contract suite.”
- Preserve existing credential routing, same-origin protections, asset permissions, job receipts, cancellations and optional source-audio omission behavior. Draft PRs only; no merges, releases, deployments or new credentials.
- Runtime defaults from the design: 15-minute metadata freshness, 5-second fetch deadline, 64 KiB response cap, 8 KiB guide-text cap; conditional ETag refresh. Credential/endpoint changes invalidate their cache partition.
- Audio has an independent 33,554,432-byte limit, not part of the image allowance. Server decoding and availability remain authoritative; consumers never certify lip-sync or GPU readiness.

## Review Focus

1. A saved audio asset has been deleted, its MIME is wrong, or an in-app caller supplies raw bytes: reject locally without generation and preserve the intended recipe. Covered by Task 5.
2. A metadata response is stale, malformed, too large, cross-origin, or from different credentials: never apply it as current executable capability. Covered by Task 3.
3. A model/schema changes while a turn is selecting a tool: finish against the pinned advertised snapshot and refresh next turn. Covered by Task 5.
4. Audio exactly covers requested seconds but misses LTX's extra frame, or H3 receives LTX defaults: retain precise server duration semantics, omit model-inapplicable fields, and do not silently pad audio. Covered by Tasks 4, 6 and 7.
5. Discovery succeeds but source-audio nodes disappear before submission, or POST acceptance is ambiguous: surface the original failure/receipt and never replay generation. Covered by Tasks 4 and 7.

## Scope and verified reconciliation

The new backend field is exactly `source_audio`, one multipart file on `POST /v1/videos`, for `ltx-2.5` and `minimax-h3`. It accepts WAV/MP3/FLAC/Ogg/M4A, normalizes the first stream to 48 kHz stereo 16-bit PCM WAV, starts at zero, trims excess, and rejects a short source. LTX requires `(seconds*24+1)/24`; H3 requires `frames/24`. It conditions generation and muxes the trimmed normalized source instead of model-generated sound. This is not Suno generation, selectable upstream `fully_copy`/`reference` modes, byte-identical copying, or proven lip-sync.

Current repository source availability is exposed by `source_audio.available`, with `supported`, MIME types, byte limit, duration policy, output/normalization, `validation: graph_contract_tested`, and `gpu_executed: false`. H3 also adds `profiles.ia2v_fl8`. A missing decoder/node prevents execution. Root live read-only checks at 08:45–08:46 UTC on 12 October still found the public deployment advertising the old contract: LTX omitted source_audio, and H3 listed it as unsupported without ia2v_fl8. Treat the deployed capability response as authoritative until that changes; no successful live source-audio generation is claimed. Existing metadata has no portable digest/version envelope yet; the new endpoint is proposed work.

First-phase generated coverage is deliberately limited to source-audio and the new metadata envelope. Existing image/video schemas remain handwritten until their separate migration; do not claim all fields are already DRY. This plan includes one CLI flag because its transport already exists on PR #2. It does not add a new Suno CLI or migrate all video constraints.

Add this reconciliation as a note in the plan and in the eventual guide records. Leave the merged historical design intact. The runtime H3 guide must explain the real soundtrack behavior without inventing audio labels, upload positions, strength controls, or unsupported reference modes. The existing reviewed Qwen/H3 prose is the initial guide content. LTX metadata must report guidance unavailable until an official LTX guide is separately verified; it must not borrow H3 syntax.

## File ownership and interfaces

**Easel:** new `easel/contracts/source_audio.py` owns `SourceAudioPolicy` and `SOURCE_AUDIO_POLICY`; new `easel/contracts/metadata.py` owns the versioned metadata envelope; new `easel/model_metadata.py` derives deployment metadata; new `easel/model_guides/*.json` holds curated content; new `scripts/export_contract.py` writes `contracts/manifest.json`, `contracts/source-audio.schema.json`, and `contracts/metadata.schema.json`. Existing `easel/source_audio.py`, `video_controls.py`, `h3_controls.py`, `app.py`, and `api_docs.py` consume those definitions. No graph rewrite.

**Client/MCP:** new `scripts/generate-easel-contract.cjs`, `contracts.lock.json`, committed `contracts/` snapshots, and `packages/media-mcp/src/generated/easel-contract.ts` hold deterministic generated interfaces/descriptors. New `source-audio-upload.ts` performs bounded binary adaptation; new `model-metadata.ts` and `metadata-cache.ts` implement discovery. Existing `video-input.ts`, `video.ts`, `server.ts`, `media-http.ts`, and desktop `src/media-reference-tools.js`, `media-mcp-client.js`, `chat-service.js`, `easel-tool-host.js` provide narrow integration points.

**CLI:** new `scripts/generate_easel_contract.py`, `contracts.lock.json`, committed `contracts/` snapshots, and `src/easel_cli/generated/source_audio.py`; new `src/easel_cli/model_metadata.py`; changes to current `client.py`/`cli.py`. Base on `feat/agent-generation-controls-20261003` at `cc5aeaf0` (PR #2), stacked on PR #1 `feat/current-api-video-jobs` at `740993d5`. Recheck heads first and use merged successors if applicable; do not rebuild from old main. Leave licensing PR #3 untouched.

### Task 1: Make the enforced source-audio policy exportable

**Files:** create `easel/contracts/__init__.py`, `easel/contracts/source_audio.py`, `scripts/export_contract.py`, `contracts/manifest.json`, `contracts/source-audio.schema.json`; modify `easel/source_audio.py`, `easel/api_docs.py`; test `tests/test_source_audio_contract.py` and existing `tests/test_source_audio.py`.

**Interfaces:** `SourceAudioPolicy` is a frozen, extra-forbid Pydantic model; `SOURCE_AUDIO_POLICY` is its sole instance. `export_contract(output_dir: Path) -> dict` writes canonical sorted JSON and returns manifest hashes. Export includes contract version `1.0.0`, exact model IDs, wire field `source_audio`, optional binary type, independent byte bound, accepted MIME aliases, normalization, zero start, two model duration formulas, and fixed serializer strategy `multipart_audio_file_v1`. The schema extension `x-easel-upload` references those policy values. Keep existing `MAX_AUDIO_BYTES`/MIME exports as derived compatibility aliases for tests/imports.

- [ ] Write failing `test_policy_matches_shipped_contract`: assert `max_bytes == 33554432`, `field == 'source_audio'`, `output == 'trimmed_source_audio'`, models are LTX/H3, and all current MIME aliases remain. Assert policy/schema/runtime discovery agree and unknown policy fields fail.
```python
def test_policy_matches_shipped_contract():
    assert SOURCE_AUDIO_POLICY.max_bytes == 33554432
    assert SOURCE_AUDIO_POLICY.field == "source_audio"
    assert SOURCE_AUDIO_POLICY.output == "trimmed_source_audio"
```

- [ ] Run `uv run --group dev pytest -q tests/test_source_audio_contract.py`; expect missing-module failures before implementation.
- [ ] Implement the immutable policy, adapt existing decoder constants and `source_audio_capabilities()` to it, and export the binary field descriptor without changing actual request parsing/graph behavior. Export checks use content hashes, not current time or git HEAD, so regeneration is deterministic.
- [ ] Run `uv run --group dev pytest -q tests/test_source_audio_contract.py tests/test_source_audio.py tests/test_openapi.py`; expect all pass with ffmpeg/ffprobe installed and no decoder skips. Export twice into temporary directories and assert identical bytes; changing policy without regenerating must fail the drift test.
- [ ] Commit this independently reviewable server refactor and generated exports with `git add` restricted to the listed paths; commit message `refactor: centralize source audio contract`.

### Task 2: Serve bounded model metadata and curated guides

**Files:** create `easel/contracts/metadata.py`, `easel/model_metadata.py`, `easel/model_guides/qwen-image-2.1.json`, `easel/model_guides/minimax-h3.json`, `contracts/metadata.schema.json`; modify `easel/app.py`, `easel/api_docs.py`, `scripts/export_contract.py`; test `tests/test_model_metadata.py`.

**Interfaces:** `ModelMetadata` has `schema_version: 1`, `model_id`, `contract_version`, `contract_digest`, `operation_ids`, `capabilities`, and `guides`. Each guide has exact model/mode/guide ID, revision, bounded snippet, sources, checked date and provenance kind. `build_model_metadata(request: Request, model_id: str) -> ModelMetadata` derives capabilities using existing discovery; `GET /v1/models/{model_id}/metadata` is authenticated, supports ETag/304, rejects unknown/unconfigured IDs, and performs GET-only backend discovery. `/v1/models` adds a same-origin relative metadata path. Unsupported guide modes return an explicit unavailable entry.

- [ ] Write failing `test_metadata_is_bounded_read_only_and_model_specific`: assert 401 without auth, 404 for unknown model, no queue/upload/submission, source-audio availability/evidence preserved, ETag/304, and guide association never crosses model/mode boundaries. Test encoded JSON <=65536 bytes and each snippet <=8192 UTF-8 bytes.
```python
def test_metadata_is_bounded_read_only_and_model_specific(audio_api):
    client, backend = audio_api
    result = client.get("/v1/models/minimax-h3/metadata")
    assert result.status_code == 200
    assert result.json()["model_id"] == "minimax-h3"
    assert len(result.content) <= 65536
    assert backend.queue_calls == 0 and not backend.uploaded
```

- [ ] Run `uv run --group dev pytest -q tests/test_model_metadata.py`; expect endpoint/model failures.
- [ ] Implement the endpoint and envelope; populate Qwen/H3 records from the reviewed companion, including corrected dialogue syntax. Include the verified source-audio behavior note but no fabricated upstream reference-control mapping. Generate the envelope schema. Keep unavailable runtime capabilities explicit rather than turning exceptions into `available:true`.
- [ ] Run `uv run --group dev pytest -q tests/test_model_metadata.py tests/test_source_audio.py tests/test_openapi.py`; expect passing assertions, including unsupported LTX guide status and coherent ETag changes when guidance changes.
- [ ] Commit `feat: expose versioned model metadata and guides`, limited to this task's files.

### Task 3: Generate client definitions and implement safe metadata caching

**Files:** create client `contracts.lock.json`, `contracts/source-audio.schema.json`, `contracts/metadata.schema.json`, `contracts/manifest.json`, `scripts/generate-easel-contract.cjs`, `packages/media-mcp/src/generated/easel-contract.ts`, `packages/media-mcp/src/model-metadata.ts`, `packages/media-mcp/src/metadata-cache.ts`, `packages/media-mcp/test/model-metadata.test.ts`, `packages/media-mcp/test/generated-contract.test.ts`; modify `packages/media-mcp/src/media-http.ts`, root `package.json`.

**Interfaces:** generated `SourceAudioInput = {data: string; mimeType: string; name?: string}`, `SourceAudioInputSchema`, `SOURCE_AUDIO_DESCRIPTOR`, `ModelMetadataSchema` and `ModelMetadata` come from locked exports. `createModelMetadataClient({cache, fetchImpl, now})` exposes `getSnapshot(provider: ProviderOptions & {model: string}, {force?: boolean, signal?: AbortSignal}) -> Promise<MetadataSnapshot>`; snapshot contains metadata or unavailable, revision, stale flag and reason. `MetadataCache` exposes `get(key)`, `set(key, entry)`, `delete(key)`. Use bounded local JSON storage in a caller-selected cache directory, safe atomic writes and no server-selected paths. Cache key hashes endpoint/model/auth configuration; never persist raw keys.

- [ ] Write failing tests for generated parity and `test_metadata_cache_is_partitioned_and_bounded`: 304 reuse; refresh at 900000 ms; abort at 5000 ms; reject oversized body before unbounded buffering, malformed/unknown major envelope, redirects, wrong model, excessive snippets and credential echoes; credential change cannot reuse another cache partition.
Test clock/cache/fetch fakes implement Task 3's injected interfaces; pin these assertions:

```typescript
assert.equal(cachedResult.revision, firstResult.revision); // HTTP 304
assert.equal(staleResult.stale, true); // refresh failed
assert.equal(otherCredentialResult.metadata, undefined); // no partition crossover
assert.equal(fetchOptions.signal.aborted, true); // fake clock advances 5000 ms
```

- [ ] Run `npx tsx --test packages/media-mcp/test/model-metadata.test.ts packages/media-mcp/test/generated-contract.test.ts`; expect missing-import failures.
- [ ] Implement allowlisted code generation for this descriptor/envelope subset; unknown executable extensions cause generator failure. Add bounded metadata JSON transport that preserves existing HTTP security. Cache fallback may return stale descriptive guides, but marks availability unknown; no cached/current mix inside one snapshot.
- [ ] Run the two explicit test files plus `npm run build:media-mcp`; expect pass. Run `node scripts/generate-easel-contract.cjs --check`; expect no diff. Add a fixture whose policy byte limit changes and prove generated output changes without hand-editing consumers.
- [ ] Commit `feat: consume generated Easel metadata contracts`.

### Task 4: Add the standalone MCP source-audio adapter

**Files:** create `packages/media-mcp/src/source-audio-upload.ts`, `packages/media-mcp/test/source-audio-upload.test.ts`; modify `video-input.ts`, `video.ts`, `server.ts`, `video-discovery.ts`, and `packages/media-mcp/test/server.test.ts`.

**Interfaces:** `decodeSourceAudio(input: SourceAudioInput) -> {bytes: Buffer; mimeType: string; name: string}` validates canonical base64, policy size/MIME, and safe managed filename. `appendSourceAudio(form: FormData, input: SourceAudioInput) -> void` uses generated field name. `GenerateVideoInput.sourceAudio?: SourceAudioInput`; existing model-specific fields are unchanged. Expose read-only `get_model_metadata({model})` returning the snapshot including guide prose. Source-audio submission requires a successful current `discoverVideoCapabilities` check for the exact model with `source_audio.available === true`; metadata guide freshness cannot substitute for this check.

- [ ] Write failing tests asserting exactly one `source_audio` multipart file, independent image/audio byte budgets, omission when absent, strict unknown-field rejection, safe filename/MIME, H3 omission of seconds, and LTX unchanged seconds behavior. Test malformed base64 and >33554432 bytes before POST; unavailable/failed capability GET prevents POST. Include the observed old deployment: absent source_audio or H3 unsupported containing source_audio blocks the new argument but does not disable ordinary no-audio generation or retrieval. If runtime rejects after discovery, return that error without retry.
Capture production FormData through the existing injected fetch:

```typescript
assert.equal(form.getAll('source_audio').length, 1);
assert.equal(form.has('seconds'), false); // H3 input
assert.equal(postCalls, 0); // absent source_audio capability or unavailable nodes
assert.equal(postCallsAfterAmbiguousFailure, 1);
```

- [ ] Run `npx tsx --test packages/media-mcp/test/source-audio-upload.test.ts packages/media-mcp/test/server.test.ts`; expect missing new field/adapter failures.
- [ ] Extend generated-source schema composition and serialize through the explicit adapter. Preserve scalar/metadata validators and POST-once behavior. Do not decode audio with a new desktop ffmpeg dependency; the server owns stream/duration verification.
- [ ] Run the focused tests and `npm run test --workspace=@easel/media-mcp`; expect pass. Receipt-loss tests must still show one POST, even after capability refresh or 503.
- [ ] Commit `feat: submit source audio through Media MCP`.

### Task 5: Connect desktop assets and safe refresh boundaries

**Files:** modify `src/media-reference-tools.js`, `src/media-mcp-client.js`, `src/chat-service.js`, `src/easel-tool-host.js`; create `src/model-metadata-snapshots.js`; tests `test/media-reference-tools.test.js`, `test/media-mcp-client.test.js`, `test/easel-tool-host.test.js`, `test/model-metadata-snapshots.test.js`.

**Interfaces:** in-app `generate_video.sourceAudioAssetId?: string | null` replaces the upload object. Existing `resolveMediaToolArguments(name,args,readAsset,signal)` resolves it to standalone `sourceAudio` only immediately before transport. `createMetadataSnapshots({loadSnapshot})` exposes `beginTurn(turnId, providers)`, `get(turnId, modelId)`, `stage(modelId, snapshot)`, `endTurn(turnId)`; active snapshots are immutable. Pass the host-owned metadata cache directory in MCP launch configuration. Add `get_model_metadata` to the allowlist and ensure all three harnesses receive the same read-only tool.

- [ ] Write failing tests for deleted/wrong-MIME assets, raw sourceAudio rejection in desktop schema, null omission, separate 32 MiB budgets, cancellation during asset read, and saved arguments containing only asset IDs. Use existing supported saved WAV/MP3/M4A assets; standalone accepts the broader API formats without enlarging desktop import formats in this task.
- [ ] Write failing refresh test: turn A sees revision 1; a notification stages revision 2; turn A remains at 1 and turn B sees 2. Endpoint/credential change clears its cache partition. Guide-only updates cannot add an unallowlisted tool.
```javascript
assert.equal(snapshots.get('turn-a', modelId).revision, '1');
snapshots.stage(modelId, revisionTwo);
assert.equal(snapshots.get('turn-a', modelId).revision, '1');
await snapshots.beginTurn('turn-b', providers);
assert.equal(snapshots.get('turn-b', modelId).revision, '2');
assert.equal(savedArguments.sourceAudio, undefined);
assert.equal(savedArguments.sourceAudioAssetId, selectedAssetId);
```

- [ ] Run `node --test test/media-reference-tools.test.js test/media-mcp-client.test.js test/easel-tool-host.test.js test/model-metadata-snapshots.test.js`; expect failing sourceAudio/refresh assertions.
- [ ] Implement asset adaptation and between-turn pinning. Subscribe to the SDK's supported tool-list change notification, stage invalidation, and emit host notifications only when a new snapshot is active. Use existing settings invalidation. Do not overwrite imported editable skills or inject guide text as executable/system instructions; agents retrieve model-specific reference content through the allowlisted tool.
- [ ] Run the focused tests, `npm test`, and `npm run build`; expect pass, including existing video/image/audio job persistence. Commit `feat: connect saved audio and model guide snapshots`.

### Task 6: Integrate the existing CLI stack

**Files:** create CLI `contracts.lock.json`, committed `contracts/` snapshots, `scripts/generate_easel_contract.py`, `src/easel_cli/generated/__init__.py`, `src/easel_cli/generated/source_audio.py`, `src/easel_cli/model_metadata.py`, `tests/test_model_metadata.py`, `tests/test_source_audio.py`; modify `src/easel_cli/client.py`, `src/easel_cli/cli.py`, `README.md`, `tests/test_video_controls.py`, `contract/server_contract.py`.

**Interfaces:** generator emits `SOURCE_AUDIO_DESCRIPTOR` and validation helpers from the same locked exports. `EaselClient.model_metadata(model: str) -> dict` uses the bounded same-origin GET; `video create --source-audio PATH` is passed as `source_audio: Path | None` to `EaselClient.create_video(prompt: str, *, ..., source_audio: Path | None = None) -> dict[str, Any]`. Add `model-info MODEL [--guide MODE]` for model/guide JSON. Read-only CLI metadata uses cached fallback clearly marked stale; supplying audio requires current capability support. Source files must be bounded regular files, never URL strings; close them on every path.

- [ ] Write failing CLI tests for `--source-audio`, model-info/guide selection, 32 MiB boundary, missing/nonregular files, correct multipart field and close behavior. Preserve PR #2's existing LTX commands and receipt safeguards; do not promise H3 generation support if its current parser lacks the required fields.
Use the CLI transport spy to pin the wire boundary:

```python
assert audio_part[0] == "source_audio"
assert audio_handle.closed
assert post_count == 0  # capability absent on the observed old deployment
assert original_receipt["id"] == returned_receipt["id"]
```

- [ ] Run `uv run --group dev pytest -q tests/test_source_audio.py tests/test_model_metadata.py tests/test_video_controls.py`; expect missing command/argument failures on the verified feature-stack base.
- [ ] Generate constants/helpers, add bounded audio file adaptation and exact-model capability checks, and add read-only metadata commands. Keep source-audio submission within models the CLI already supports; H3 metadata/guide inspection remains available independently.
- [ ] Run focused tests, full `uv run --group dev pytest -q`, `uv build`, and installed CLI help; expect pass. Generated check `python scripts/generate_easel_contract.py --check` must be clean. Commit `feat: add source audio and model metadata to CLI`.

### Task 7: Require cross-repository contract verification

**Files:** server modify `tests/test_mcp_contract.py`, create `tests/test_contract_pipeline.py`, `scripts/check_consumers.py`, `contracts/consumers.lock.json`, `.github/workflows/contracts.yml`; client modify `.github/workflows/test.yml`; CLI modify `.github/workflows/ci.yml` and `contract/server_contract.py`.

**Interfaces:** `python scripts/check_consumers.py --mcp-source PATH --cli-source PATH` checks explicit pinned consumer refs/lock digests/build outputs, requires ffmpeg/ffprobe, and executes production serializers against actual FastAPI/fake Comfy backends. Missing inputs are fatal in this runner, even if existing optional local modules remain skippable. The lock has exact consumer commits and the contract digest for a verified candidate set; the server commit is the explicit CI checkout and is recorded in verification output, avoiding a self-referential commit lock. Each consumer lock records its exact server export commit and generator version; do not require circular source SHA equality when fixture-only CI commits change no contract digest.

- [ ] Write failing integration tests: WAV+image from actual MCP/CLI reaches the graph, normalized PCM duration is exact, original waveform is muxed, H3 semantic/temporal guidance survives MCP audio, no-audio graphs unchanged, too-short LTX/H3 audio rejects without upstream mutation, and missing/unavailable runtime fails with zero POST replay. Use fixtures/fake backends only.
Extend the real FastAPI/fake-backend fixtures, not a second serializer:

```python
assert response.json()["frames"] == expected_frames
assert normalized_wav.getnframes() == expected_frames * 2000
assert normalized_wav.getframerate() == 48000
assert normalized_wav.getnchannels() == 2
assert rejected_backend.uploaded == []  # short audio or incompatible runtime
```

- [ ] Run the new runner without each required input in turn; expect nonzero failure, not skip. Run with old consumer builds; expect absent-field failures.
- [ ] Implement the mandatory runner, export drift checks, and CI jobs using explicitly pinned checkouts. Run the existing CLI contract module with server dev environment and both source directories on PYTHONPATH. Use already available read permissions only; if private cross-repo checkout is unauthorized, fail with an actionable access blocker, never create secrets or loosen repository permissions.
- [ ] Run `uv run --group dev python scripts/check_consumers.py --mcp-source ../easel-client --cli-source ../easel-cli` from the server checkout; expect all contract tests passing, no skipped decoder tests and no network/generation. Run all three repos' normal suites/builds and `git diff --check` against final commits.
- [ ] Commit each repository's CI/test changes separately; publish coordinated draft PRs with candidate commit/digest matrix and exact results. Verify remote heads and CI for those heads before reporting completion. No auto-merge or deployment.

## Deferred work and execution order

This phase establishes generated source-audio propagation and metadata/guide consumption. It intentionally defers full image/video model extraction, blanket removal of old schema copies, broader desktop audio import formats, H3 CLI generation if absent, unrelated Suno migration, and unattended cross-repo PR creation. These are the design's later phases, not gaps silently claimed complete. The deterministic generator/lock/check workflow is the foundation for that later automation; no GitHub access grants are created here.

Execute Tasks 1 and 2 first. Client Task 3 and CLI generator groundwork can then proceed independently against a pinned export. Complete Tasks 4 and 5 together under the client owner; Task 6 stays on the existing CLI stack. Task 7 reviews the combined candidate set. Fresh reviewers should check server policy/runtime parity, client asset/security behavior, and CLI resource handling before the integrated gate.

## Plan self review

Coverage checked against the merged design: canonical enforcement/export (1), metadata/guides (2), generated consumer/cache safety (3), upload semantics (4), stable agent refresh/assets (5), existing CLI integration (6), mandatory pinned tests/drift checks (7). Full migration and unattended PR automation remain explicit later phases. All five Review Focus conditions have owning tests. Names and shapes for SourceAudioInput, sourceAudio/sourceAudioAssetId/source_audio, metadata snapshot and policy descriptors agree across tasks. No inferred audio mode, live generation, new credential or deployment step is present.

Recommended execution: **subagent-driven**, with one server owner, one client/MCP owner, and one CLI owner after the shared export exists, each independently reviewed before whole-stack verification. Request plan review and execution-method selection before implementation.
