# Easel shared API contract and model guidance design

Status: Design draft for review, 12 October 2026; video source-audio integration awaits the tested patch.

## Executive summary

Keep one validated contract in Easel. Generate client, MCP, and CLI field schemas and serializers from it; let runtime metadata supply changing capabilities and compact model guides. Existing secure transports, media assets, and job recovery stay in place.

Start with the source-audio input for video currently being tested locally, activate cross-repository drift tests, and prove propagation on that narrow contract. Expand to remaining image/video inputs after the pipeline works. New upload semantics still need a reviewed adapter; metadata alone cannot enable unknown operations. Consumer updates arrive as tested draft PRs, while compatible guide and capability changes refresh between turns.

The companion [model prompt guides](2026-10-12-model-prompt-guides.md) supply reviewed Qwen and H3 content and official sources. No merge, deployment, release, or new credentials are included.

## Intent and success

Define generation inputs once in Easel and propagate them automatically to the desktop client, standalone Media MCP, and CLI. Give agents short, current, model-specific prompting guidance from official sources without copying large reference documents into every consumer.

The proposed hybrid keeps runtime discovery for changing model availability, limits, and guide text, and uses deterministic build-time generation for typed contracts and serialization. It preserves the existing transport security, asset handling, accepted-job recovery, and user workflows. Success means a compatible field change produces reviewed consumer updates without separately rewriting field schemas and mappings; a model or guide metadata update appears after a bounded refresh without an app rebuild.

The pending audio reference is source audio conditioning a video request, not the existing Suno music/speech/sound generation API. Its exact pushed revision, fields, upload semantics, and tests must be inspected before selecting the first extraction. This specification does not infer new source-audio fields or require an unrelated Suno migration.

## Verified starting point

The read-only audit inspected Easel main at `68d6b239`, easel-client main at `d8dda0f1`, and easel-cli main at `ba2400ce`. Recheck these refs before implementation.

- [Easel API documentation](https://github.com/venetanji/easel/blob/68d6b2392d0200d6c96ccce5327d7c93060ae55d/easel/api_docs.py) is explicitly documentation-only. Image schemas duplicate manual parsing. The video OpenAPI description warns that its form omits H3-specific inputs and can send inappropriate LTX defaults. It is not yet a complete generation source.
- [Audio requests](https://github.com/venetanji/easel/blob/68d6b2392d0200d6c96ccce5327d7c93060ae55d/easel/audio_api.py) already use Pydantic models. Video capabilities come from `easel/video_controls.py` and `easel/h3_controls.py`.
- [Media MCP audio](https://github.com/venetanji/easel-client/blob/d8dda0f1b953136ee7fc827eeeb39104471770b4/packages/media-mcp/src/audio.ts) repeats strict schemas, wire-name mappings, and cross-field validation. `server.ts`, `video-input.ts`, `video.ts`, and `video-discovery.ts` repeat other contract details. `src/media-reference-tools.js` deliberately adapts uploads to desktop asset IDs.
- [Cross-language contract tests](https://github.com/venetanji/easel/blob/68d6b2392d0200d6c96ccce5327d7c93060ae55d/tests/test_mcp_contract.py) already exercise actual MCP payloads against FastAPI and assert audio field coverage. The module skips without `EASEL_MEDIA_MCP_SOURCE`. [Client CI](https://github.com/venetanji/easel-client/blob/d8dda0f1b953136ee7fc827eeeb39104471770b4/.github/workflows/test.yml) does not activate it.
- The desktop caches tool descriptors in `src/easel-tool-host.js`; `src/main.js` invalidates them after settings/model refresh. `src/media-mcp-client.js` has a fixed tool allowlist and no tool-list notification subscription. Preserve the allowlist.
- [Bundled guidance](https://github.com/venetanji/easel-client/blob/d8dda0f1b953136ee7fc827eeeb39104471770b4/.agents/skills/easel-media/SKILL.md) contains copied model facts and adapter lists, including stale H3 limitations. Imported skills are editable local copies and do not automatically refresh.

## Canonical ownership and generated artifacts

Keep the canonical contract in `venetanji/easel`, beside the implementation. Do not introduce a fourth repository or a separately maintained universal SDK in the first phase.

Proposed source organization:

- `easel/contracts/`: actual validated request/response models and a small operation registry. Extract the smallest authoritative video request variant after inspecting the source-audio patch; preserve existing imports/behavior. Existing Suno Pydantic models are a useful precedent, not the first-phase target. Other image/video parsing is migrated incrementally into equally authoritative models. Model-dependent constraints that need runtime state remain server validation, backed by tests.
- `easel/model_guides/`: reviewed compact guidance records, keyed by exact provider model ID and workflow, with official source URLs, checked date, guide revision, and clearly separate Easel-specific caveats.
- `scripts/export_contract.py`: deterministic export of supported models/operations into `contracts/openapi.json`, `contracts/manifest.json`, and reusable JSON Schemas. Documentation derives from the same validated models. The manifest records contract version, source revision, hashes, generator requirements, and stable operation identifiers.
- Generated consumer files in `easel-client/packages/media-mcp/src/generated/` and `easel-cli/src/easel_cli/generated/`, with a checked-in `contracts.lock.json` per consumer recording source revision, digest, and generator version. No manual edits to generated output.

The operation registry may describe wire aliases, content type, model discrimination, upload role, and supported serializer strategy. Those descriptors must reference validated fields rather than redeclare their bounds. Plain field constraints are generated; server-only and cross-field rules must be explicitly represented or tested, never silently claimed to be enforced by a generated schema.

Generate TS/Python types, validation schemas, ordinary field mappings, and CLI option descriptions from the export. Keep handwritten wrappers for credential routing, job receipts/polling, filesystem output, and upload-to-asset conversion. Maintain existing camelCase MCP names and CLI flags through explicit compatibility aliases. Improve the current API schemas before pointing a generic OpenAPI generator at them.

## Runtime metadata and compact model guides

Add a versioned, authenticated model metadata endpoint, proposed as `GET /v1/models/{model_id}/metadata`, discoverable from each `/v1/models` entry. It returns data for an exact configured model: metadata schema version, contract version/digest, supported operation IDs, deployment-specific limits/capability evidence, and a compact guide with its own revision. Existing video capabilities remain backward-compatible and derive from the same definitions. Native Suno model IDs remain distinct from Easel routing IDs.

Metadata contains no credentials, host filesystem paths, executable content, arbitrary operation URLs, or automatically followed external references. Source links are attribution, not instructions to fetch or execute content. Consumers allow only known operation identifiers and same-origin metadata requests through their existing authenticated transport. Unknown schema keywords or serializer strategies cannot enable execution.

Guides contain a concise model/workflow summary, prompt recipe, reference behavior, do/don't examples, and source provenance. Keep hard API bounds in schema data; guide prose links to them rather than repeating numerical tables. Treat official recommendations, Easel restrictions, and local experimental observations as different evidence categories. No automatic scraping is required at runtime. The initial Qwen and H3 guide set and verified official sources are in the [companion guide specification](2026-10-12-model-prompt-guides.md). Further model guides use the same provenance requirements.

Refresh on explicit model refresh and at the next turn/session boundary when the cache is stale. Proposed defaults: one conditional GET per selected model per 15 minutes, a 5-second request deadline, 64 KiB maximum metadata response, and 8 KiB maximum guide text. Cache by endpoint identity, selected model, and authenticated configuration identity without storing raw credentials in cache keys. Use ETag/If-None-Match where available and reject cross-origin redirects. Changed credentials or endpoints invalidate their cache partition.

Pin a coherent metadata/schema snapshot for each active agent turn. Stage updates for the next turn so an advertised schema cannot change between tool selection and execution. Settings/model refresh continues to invalidate host descriptors; long-lived MCP connections additionally receive the supported tool-list change notification after the new snapshot becomes active. The desktop subscribes and relists at its next safe boundary. No mid-submission retry or mutation is caused by a refresh.

## What updates automatically

Runtime updates can change guide text, availability evidence, and values permitted by an already implemented operation/serializer. First-phase tool schemas expose only the intersection of server-advertised support and the consumer's generated contract. Discovery never makes an unsupported operation callable merely because the server lists it.

New compatible fields trigger regeneration and a consumer update PR. Once that consumer version is installed, the field is exposed consistently in validation, help/tool schema, and serialization. This is automatic source propagation, not a promise that every new field instantly appears in already installed binaries.

Fully dynamic arbitrary field exposure is deferred. A later declarative extension mechanism could expose new scalar fields within pre-approved operation strategies, but it is unnecessary for the initial design and must retain strict bounds and validation. New upload types, reference indexing, request content types, authentication behavior, operations, or lifecycle semantics always require a reviewed adapter change and tests. Unknown inputs are rejected clearly, never dropped, guessed, or converted to a different generation recipe.

## Compatibility and offline behavior

Version the portable contract, runtime metadata envelope, and guide content separately. Optional fields with unchanged existing meaning are contract-minor changes; removed/renamed fields, changed defaults or semantics, new required inputs, and incompatible unions are major changes. Enum expansion is assessed against generated validators and exhaustive consumers rather than assumed safe. Source/digest checks distinguish an actual contract change from prose-only guidance updates.

Consumers accept only supported major versions. New descriptive metadata can be ignored safely; unknown executable semantics cannot. They never submit optional fields absent from the selected deployed server's supported contract. Rollback selects the last tested consumer lock/export pair and retains accepted receipts independently of model retirement or contract revision.

If metadata is unavailable, use a validated same-endpoint cache for descriptive guidance with a visible stale indicator. Cached runtime availability is not current readiness evidence. Without a compatible cache, use the bundled generated contract only for existing known operations and show that discovery is unavailable; do not expose new fields or claim backend readiness. Failure of generation or retrieval is reported normally. A metadata failure never triggers generation, a settings change, or automatic POST replay.

## Automated propagation and CI

Each server contract change regenerates exports and fails CI on an uncommitted difference or incompatible unversioned change. Consumer CI reproduces generated output from its locked source and rejects drift. Cross-repo checks use explicitly pinned source revisions, never whatever happened to be on main that day.

The consumer update flow detects an accepted contract revision, regenerates outputs, updates the lock, runs tests, and opens or updates one draft PR per repository/revision stream. It deduplicates existing sync branches and adds follow-up commits to its own work branch only. It never overwrites human edits or merges/releases/deploys. Use currently authorized repository access only. A repository's default CI token must not be assumed to have cross-repo access; if the existing automation identity lacks access, leave a verified artifact and report the access blocker. Creating a PAT, GitHub App grant, secret, or broader persistent permission is outside this design's authorization and requires a separate decision.

Mandatory checks for touched operations:

1. Deterministic export/generation and no generated-file drift; positive/negative request fixtures for every model variant.
2. Actual generated MCP/CLI serialization submitted to real FastAPI with fake backends; exact accepted fields, defaults, enums, decimal-string uint64 seeds, content types, ordered multipart references, and rejection before upload on invalid inputs.
3. Activate `tests/test_mcp_contract.py` with a pinned built client checkout and make missing checkout/build a CI failure, not a skip. Add equivalent pinned CLI contract coverage using its existing feature-stack tests.
4. Full relevant consumer tests, type checks/builds, and packaged command/tool discovery. Existing safety tests remain required: same-origin credentials, download limits, accepted receipt persistence, no automatic generation retries, and cancellation semantics.
5. Metadata cache/version/ETag tests, stale/offline fallbacks, malformed or oversized data, unknown operation strategies, credential partitioning, and schema changes staged between turns. Guide tests verify exact model association, source provenance fields, and bounded size.

No GPU generation, Suno charges, deployment or live-secret testing is required for this CI contract suite. Fake-backend success does not establish visual quality or live readiness.

## Existing work and integration order

- Easel's only audited open PR was [#7](https://github.com/venetanji/easel/pull/7), `fix/swagger-audio-download-guidance` at `bd071256`, targeting `main` at `68d6b239`. Coordinate documentation edits rather than duplicate it. The owner's local video source-audio patch is an additional pending input, not a known published branch.
- easel-client had no open PRs at audit time. Its main already handles Suno audio and H3 semantic/temporal uploads. Preserve those paths and remove stale guidance only after replacement metadata is available.
- CLI [#1](https://github.com/venetanji/easel-cli/pull/1), `feat/current-api-video-jobs` at `740993d5`, targets `main` at `ba2400ce`.
- CLI [#2](https://github.com/venetanji/easel-cli/pull/2), `feat/agent-generation-controls-20261003` at `cc5aeaf0`, targets `feat/current-api-video-jobs` at `740993d5`. It adds typed LoRAs/guides but still uses LTX-specific defaults/size choices. Extend this stack or its subsequently merged successor; do not recreate jobs/video on old main.
- CLI [#3](https://github.com/venetanji/easel-cli/pull/3), `chore/gpl-alignment-20261005` at `03f031c2`, targets `main` at `ba2400ce`. Preserve its licensing scope and avoid accidental conflicting metadata edits.

Re-read open PRs and exact heads before creating branches. No PR merge is authorized by this document.

## Phased rollout and acceptance

**Phase 1: small shared-contract foundation.** Inspect the pushed, tested video source-audio revision and settle its real request and upload contract. Extract that smallest authoritative video variant into shared validated definitions without changing behavior; generate its existing/new supported MCP field schemas and ordinary mappings while preserving explicit upload adapters. Activate mandatory cross-repo tests. Add bounded metadata/guide discovery and cache to the shared Media MCP provider with desktop refresh integration, and read-only CLI metadata/guide inspection on the existing stack. The current Suno models demonstrate the pattern but need no unrelated migration. Acceptance: the tested source-audio request reaches FastAPI through the consumer with its intended reference binding intact, schema drift fails CI, and a guide revision refreshes between turns with an offline fallback.

**Phase 2: complete typed propagation.** Move the remaining image/video variants onto validated canonical models; generate their field schemas/serializers/options; align CLI coverage with the approved video source-audio contract and preserve existing commands. Migrate existing Suno request metadata only where it removes confirmed duplication without expanding product scope. Add the draft consumer-update PR flow with verified existing access. Acceptance: a compatible optional field change produces tested consumer PRs without manually editing each schema/mapping; old deployed servers remain compatible.

**Phase 3: remove redundant copies.** Replace repeated capability bounds and copied model guides with generated/runtime references, keeping user-edited skills intact. Keep small general creative skills and safe asset/lifecycle adapters handwritten. Acceptance: consumer source has no competing editable definition of migrated wire fields; model-specific factual guidance has one reviewed home and a documented refresh boundary.

Each phase ends with evidence from tests and reviewed draft changes. The next step is to reconcile the pushed video source-audio patch, review this design and its companion guides, then prepare the implementation plan.
