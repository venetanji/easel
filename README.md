# Easel

An OpenAI-compatible image and video API gateway for a separately operated
[ComfyUI](https://github.com/Comfy-Org/ComfyUI) service, with audio generation
through a separately operated Suno MCP/REST service. Python graph builders,
authenticated admission, managed uploads and durable image job receipts live here.

## Develop and run

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required:

```sh
uv sync --frozen
cp .env.example .env
# Configure the ComfyUI endpoints and authentication described in .env.example.
uv run uvicorn easel.app:app --env-file .env --host 0.0.0.0 --port 8080
```

The application reads environment variables; the command above loads `.env`
through Uvicorn. The supplied Compose configuration also reads it. Keep credentials
out of version control. API and model setup details:

- [Image jobs](IMAGE_JOBS.md)
- [Video API](VIDEO_API.md) and [video LoRAs](VIDEO_LORAS.md)
- [Model serving](MODEL_SERVING.md)
- [Audio API](AUDIO_API.md)
- [Graph provenance](GRAPH_PROVENANCE.md) and [H3 baseline](H3_BASELINE.md)

Tests use fake backends by default and do not submit GPU jobs:

```sh
uv run --frozen pytest
uv build
```

Live ComfyUI integration and cross-language client tests are explicit opt-ins;
see `tests/test_integration.py` and `tests/test_mcp_contract.py`. No model weights
or ComfyUI runtime are included in the Python package.

## License and third-party materials

Copyright (c) 2026 Venetanji. Easel-owned code, including the maintainer's Python
graph generators, is licensed under **GPL-3.0-or-later**: GNU GPL version 3 or,
at your option, any later version. The program comes without warranty. See the
full [LICENSE](LICENSE) and scoped [NOTICE](NOTICE).

Third-party material keeps its existing terms. The `licenses/` directory retains
creative-skills and official workflow-template MIT notices and ComfyUI's GPLv3
text for referenced/adapted upstream material and captured schema descriptions.
The project's SPDX field states the own-code grant; it does not relicense
upstream portions, revoke earlier MIT grants, or expand ComfyUI's permissions.
Model weights, LoRAs and other model assets have separate terms.

## Source and redistribution

The preferred application source is this repository, including `easel/`,
`pyproject.toml`, `uv.lock`, `Dockerfile`, `compose.yaml`, `scripts/`, tests,
fixtures and documentation. For a wheel/container release, record the exact
source commit and provide the corresponding source and required dependency
materials through an applicable GPL section 6 route, with clear source-access
directions alongside the binary download. A moving branch link alone does not
identify the source used to build a particular artifact.

The Dockerfile copies the README and license materials before package building
and retains application Python source. That is not, by itself, proof that every
container redistribution obligation is satisfied. Before publishing an image,
record the base-image digest, inspect installed dependency/OS notices and source
requirements, and reconcile them with the shipped layers. See the
[licensing audit](docs/license-audit.md) for the scope and verification limits.
