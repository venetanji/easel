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
uv run uvicorn easel.app:app --env-file .env --host 0.0.0.0 --port 8080 --no-proxy-headers
```

The application reads environment variables; the command above loads `.env`
through Uvicorn. The supplied Compose configuration also reads it. Keep credentials
out of version control. API and model setup details:

- [Image jobs](IMAGE_JOBS.md)
- [Video API](VIDEO_API.md) and [video LoRAs](VIDEO_LORAS.md)
- [Video sizing, resource limits and opt-in GPU benchmarks](VIDEO_SIZING.md)
- [Model serving](MODEL_SERVING.md)
- [Audio API](AUDIO_API.md)
- [Graph provenance](GRAPH_PROVENANCE.md) and [H3 baseline](H3_BASELINE.md)

Tests use fake backends by default and do not submit GPU jobs:

```sh
uv run --frozen pytest
uv build
```

An opt-in bounded concurrency audit submits real generations (including paid
Suno requests), saves receipts, and never retries a generation POST:

```sh
uv run --frozen python scripts/audit_capacity.py --live \
  --url https://easel.ait4x.org --output /tmp/easel-capacity.json
uv run --frozen python scripts/audit_capacity.py --resume \
  --url https://easel.ait4x.org --output /tmp/easel-capacity.json
```

The default burst is 12 image requests, 12 one-second video requests, and two
Suno music requests. Each GPU backend admits at most eight queued/running jobs;
overflow is expected to return 429. Only run during an agreed idle window, keep
the report, and use `--resume` after a disconnect instead of starting a new
burst. `--submit-only` stops after saving acceptance responses, for a controlled
Easel restart followed by receipt-only polling. This does not make ComfyUI's
own queue/history durable; see `docs/CAPACITY_AUDIT_2026-10-09.md` for measured
limits and recovery gaps.

Live ComfyUI integration and cross-language client tests are explicit opt-ins;
see `tests/test_integration.py` and `tests/test_mcp_contract.py`. No model weights
or ComfyUI runtime are included in the Python package.

## Authentication and local HTTP

With `EASEL_API_KEY` set, loopback and Headscale/Tailscale clients are already
authenticated by default: `127.0.0.0/8`, `::1/128`, `100.64.0.0/10` and
`fd7a:115c:a1e0::/48`. They can call the API over HTTP without a bearer token;
other clients still need `Authorization: Bearer <token>`. This policy applies to
image, video, audio, job status and content routes. An unset API key still disables
authentication for everyone.

`EASEL_TRUSTED_CLIENT_CIDRS` replaces the default comma-separated allowlist. Set it
explicitly empty to require bearer auth from every client. Use your actual
Headscale address prefixes if they differ from the defaults; do not trust all
private/LAN subnets just to accommodate local agents.

Run Uvicorn with `--no-proxy-headers` (as the Docker image does) so authentication
starts from the socket peer, not a caller-supplied header. If a reverse proxy is
in front of Easel, put its exact backend-facing IP/CIDR in
`EASEL_TRUSTED_PROXY_CIDRS`. Only those peers may supply `X-Forwarded-For`;
Easel walks the chain from right to left, skips configured proxies, and checks
the first non-proxy address against the client allowlist. Missing or invalid
forwarded addresses require a bearer token. Proxies must append the actual
client IP or overwrite the header, never pass an unvalidated header unchanged.
Configure even loopback/tailnet proxies this way so their public clients do not
inherit the proxy's own trusted-client status.

Public proxies on the tailnet need an explicit exception to client trust:
set `EASEL_REQUIRE_AUTH_CIDRS=100.64.0.11/32,fd7a:115c:a1e0::b/128` for the
current ait4x/Caddy node, covering both its IPv4 and IPv6 tailnet addresses.
These addresses always require a bearer token, even when they also match a
trusted-client or trusted-proxy CIDR. The exception is checked against both the
socket peer and each forwarded hop traversed through a trusted proxy, so Caddy
cannot grant public traffic the tailnet bypass. Keep this list updated when a
public proxy's tailnet address changes. Direct requests from other tailnet nodes
retain their existing token-free access.

If a proxy rewrites the Host header (as the Headscale Serve sidecar does), set
`EASEL_PUBLIC_URL=https://easel.ait4x.org`. Bearer-authenticated external clients
then receive reachable HTTPS status/content URLs instead of the internal tailnet
origin. Trusted local/tailnet clients keep their original HTTP URLs. The setting
is an origin only, without a `/v1` suffix; no forwarded Host/Proto header is
trusted to construct these public URLs.

With Docker, `http://127.0.0.1:8799/v1` may arrive from a bridge gateway rather
than loopback. Before adding that exact gateway `/32` to the client allowlist,
set `EASEL_BIND_HOST=127.0.0.1` so Docker cannot also hide a public IPv6 client's
address behind the gateway. A Headscale Serve sidecar can still reach the backend
on its Docker network: add its exact Docker IP to the proxy allowlist and use
`http://<easel-headscale-name>/v1` from Headscale nodes. Keep proxy addresses
stable and update the allowlist if a sidecar is recreated with a different IP.
No HTTPS redirect or login roundtrip is needed on either path.

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
