# Agent generation controls implementation plan

> For agentic workers: use test-driven changes and independent whole-branch review.

Goal: expose one validated generation contract through Easel, CLI and both agent harnesses.
Architecture: keep graph creation in Python; typed clients serialize validated metadata and bounded uploads.
Spec: ../specs/2026-10-03-agent-generation-controls-design.md

## Ownership and order

1. Coordinator: server capability/guide parser, API integration, graph construction, server tests and public docs. Write failing graph and admission tests, implement, run full non-live server suite. Wait for LTX-2.5 guide semantics review before graph changes.
2. CLI worker: src/easel_cli/{client,cli}.py, own tests, contract/server_contract.py and README. Add typed selections/discovery and serialized guiding uploads; preserve receipt/resume. Run existing tests then expanded contract suite against coordinator server.
3. MCP worker: packages/media-mcp/**, src/media-reference-tools.js, src/media-mcp-client.js, src/harness-instructions.js and corresponding focused tests. Typed discovery, advanced options, safe asset bridge, exact uint64 seed. Run transport/schema/provider tests and compile.
4. Coordinator: update shipped easel-media skill/reference docs and canonical creative-skills API guidance; extend opt-in local-key runner without paid requests. Ensure docs match actual schemas.
5. Independent reviewer: server graph/crop order; precise wire contract; unsafe/ambiguous inputs; endpoint routing and credentials; no dropped options; saved IDs on error/resume; guide validation labeling. Fix findings, rerun final aggregate suites and existing offline export integration.
6. Publish linked draft PRs, verify remote trees and exact-head CI. Do not merge or deploy. Existing editor/CLI drafts remain preserved; new branches start from their remote heads.

## Review focus

- Guide latent metadata and crop outputs cannot leak extra frames into upscale/decode.
- Numeric seed must survive JavaScript beyond 2**53 without rounding.
- Invalid guide/file/LoRA combinations must fail before upstream upload or POST.
- Discovery must distinguish code support, installed assets/nodes and live execution evidence.
- Interrupted accepted jobs retain original IDs and never trigger replacement generation.
