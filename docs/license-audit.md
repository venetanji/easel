# GPL alignment provenance audit (5 October 2026)

Status: **Easel-owned code declared GPL-3.0-or-later, with upstream terms
preserved**. This supersedes the earlier ownership-confirmation blocker.
The maintainer confirmed on 5 October 2026 that Comfy Graph recipes derive from
official ComfyUI documentation and GitHub workflows, and that the Python
generator is their own GPL code. This is the ownership basis for the own-code
grant; Git author names alone are not that basis.

## Audited material and disposition

Fresh main `361aa533a674fcdf2d4fe0d8f3682ce4ae11ea2d` contained 52 tracked files,
with no root LICENSE or pyproject license declaration. The alignment adds an
unmodified GPLv3 text, an explicit own-code GPL-3.0-or-later grant in NOTICE,
README/package metadata, retained upstream licenses and regression tests.

| Component | Evidence | Disposition |
|---|---|---|
| Easel-owned API, configuration, jobs, validation, tests and Python generators | Maintainer confirmation; source and history | GPL-3.0-or-later own-code grant |
| `easel/flux_graph.py`, `easel/workflow_graph.py` | Explicitly vendored/adapted from creative-skills Flux/helper code | Preserve MIT copyright/grant; record adapted files and modifications |
| `easel/video_graph.py` | Owner-created generator; official workflow-derived recipes; GRAPH_PROVENANCE.md | Own-code GPL grant; retain applicable upstream terms; canonical LTX-2.5 lineage remains a technical provenance limit |
| `easel/h3_graph.py`, `tests/fixtures/h3/h3_*_prototype.json` | Owner-created generator/recipes; pinned official H3 sources in H3_BASELINE.md | Own-code GPL grant; keep official source terms on upstream portions; do not claim byte-identical upstream prototypes |
| `tests/fixtures/h3/runtime_nodes.json` | Filtered 4 October capture at pinned ComfyUI revision | Preserve provenance and GPLv3 text for copyrightable copied schema descriptions; no broad claim that node names/interfaces are copyrightable |
| Python dependencies | pyproject.toml and uv.lock; no dependency tree vendored | Dependencies retain their own terms and notices; actual release inventory still required |
| ComfyUI runtime | Separate HTTP service; not copied by Dockerfile | Independent upstream terms; own-project grant does not relicense it |
| Model weights / LoRAs / encoders / VAEs | Pinned model metadata and download helper, no weights tracked | Model-specific terms stay separate; application license grants no blanket asset permissions |

## Source evidence

The inspected creative-skills source is
[`67f185c8f95182690e1ac56e734aa649f71b4df6`](https://github.com/venetanji/creative-skills/tree/67f185c8f95182690e1ac56e734aa649f71b4df6).
Its [root LICENSE](https://github.com/venetanji/creative-skills/blob/67f185c8f95182690e1ac56e734aa649f71b4df6/LICENSE)
is MIT, copyright 2026 Venetanji, preserved verbatim as
`licenses/creative-skills-MIT.txt` (Git blob
`26daa8e909e25292eaf5a70a1b9eced041c20d9e`). The public LTX runtime there is
LTX-2.3. Owner confirmation does not turn that into evidence of importing the
unpublished canonical LTX-2.5 implementation. Preserve the distinctions in
[GRAPH_PROVENANCE.md](../GRAPH_PROVENANCE.md) and [H3_BASELINE.md](../H3_BASELINE.md).

The pinned
[workflow-template LICENSE](https://github.com/Comfy-Org/workflow_templates/blob/0e5c5efb32ba6f3365d6da07da64aaf668157042/LICENSE)
is MIT, copyright 2023-present Comfy Org, preserved verbatim as
`licenses/comfy-workflow-templates-MIT.txt` (Git blob
`ec6ae8a4821329f417a3e74e6dd108f499d08290`). The [H3 R2V](https://github.com/Comfy-Org/workflow_templates/blob/0e5c5efb32ba6f3365d6da07da64aaf668157042/templates/video_minimax_h3_r2v.json)
and [multiframe](https://github.com/Comfy-Org/workflow_templates/blob/0e5c5efb32ba6f3365d6da07da64aaf668157042/templates/video_minimax_h3_multiframe_reference.json)
workflows are official reference sources, not asserted byte-identical origins
of every local prototype. Installed workflow package versions and hashes remain
recorded separately in H3_BASELINE.md.

ComfyUI's [H3 revision LICENSE](https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/LICENSE)
and [LTX revision LICENSE](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/LICENSE)
have the same Git blob `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7`, the full GPLv3
text preserved as `licenses/ComfyUI-GPL-3.0.txt`. The H3 revision's
[pyproject.toml](https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/pyproject.toml)
points to that file. These inspected materials do not supply a separate explicit
"or later" declaration. We preserve the actual upstream text, do not substitute
Easel's grant, and do not infer a definitive GPL-3.0-only classification from
the absence of another statement in those files. No new grant for upstream
ComfyUI content is made here.

The source evidence and maintainer confirmation resolve the earlier request to
confirm ownership of the Python generator. A complete line-by-line historical
map of unpublished recipes was not reconstructed. That limitation is disclosed
rather than treated as proof of infringement or a reason to withhold the
maintainer's own-code grant. No third-party notices are removed.

## Package and container alignment

- `pyproject.toml` uses PEP 639 SPDX metadata `GPL-3.0-or-later` for the project's
  own-code grant, scoped by NOTICE, and explicit `license-files` covering the
  full GPL, NOTICE and all retained upstream license texts. Hatchling is at least
  1.27 for PEP 639 support. This field does not replace the per-source terms.
- README describes source/build/test routes and the code/model boundary.
- Dockerfile copies README, LICENSE, NOTICE and `licenses/` before `uv sync`,
  avoiding missing metadata inputs. It keeps application Python source and
  dependency distributions; it does not copy ComfyUI or model weights.
- Before any binary/container redistribution, provide the matching preferred
  application source, exact commit, lockfile, Dockerfile, build/install scripts
  and required dependency materials under an applicable GPL section 6 route.
  A moving repository link or Python source inside an image alone is not proof
  that all source obligations are met.
- Record the actual base-image digest and OS/package inventory and reconcile
  their separate notices and source requirements with the image layers.
  Downloaded model weights must not be silently bundled under the code license.

## Verification and limits

The seven licensing tests were first run against the missing declaration and
all failed for the intended omissions; they pass after the alignment. They
cover own-code metadata, full license/notice material, retained MIT grants,
verbatim ComfyUI license hash, graph-module notice pointers, and Docker build
input ordering. Normal offline tests and wheel/sdist checks are recorded in the
PR verification report.

This source change is not a released artifact or a completed container-license
inventory. No container was built/deployed, GPU job submitted, model downloaded,
new external license accepted, or release published by this task. Container
layer/source verification remains a release-time step; no claim about existing
published artifacts is made.

Primary GPL references: [GNU GPLv3](https://www.gnu.org/licenses/gpl.en.html),
sections 1, 5, 6 and 14; [GNU FAQ](https://www.gnu.org/licenses/gpl-faq.en.html).
This is an engineering provenance review, not a legal guarantee.
