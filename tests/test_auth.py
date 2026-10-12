"""Local/tailnet authentication and reverse-proxy trust without network calls."""
import dataclasses

import httpx
import pytest

from easel.app import create_app
from easel.config import Settings
from tests.test_app import FakeComfy
from tests.test_video_loras import LoRAComfy


@pytest.fixture
def api(tmp_path):
    def client(host, env=None):
        settings = Settings.from_env({"EASEL_API_KEY": "secret", **(env or {})})
        settings = dataclasses.replace(settings, image_job_dir=str(tmp_path),
                                       comfy_video_url="http://video")
        app = create_app(settings=settings, comfy=FakeComfy(), comfy_video=LoRAComfy())
        transport = httpx.ASGITransport(app, client=(host, 123) if host is not None else None)
        return httpx.AsyncClient(transport=transport, base_url="http://easel")
    return client


@pytest.mark.parametrize("host", [
    "127.0.0.1", "127.20.30.40", "::1", "::ffff:127.0.0.1",
    "100.64.0.1", "100.127.255.254", "fd7a:115c:a1e0::7", "::ffff:100.64.0.7",
])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
async def test_local_and_headscale_clients_need_no_bearer(api, host, headers):
    async with api(host) as client:
        response = await client.get("/v1/models", headers=headers)
    assert response.status_code == 200


@pytest.mark.parametrize("host", [
    "203.0.113.10", "10.0.0.1", "172.17.0.1", "192.168.0.1",
    "100.63.255.255", "100.128.0.0", "2001:db8::1", "fd7a:115c:a1e1::7",
    "::ffff:203.0.113.10", "localhost", None,
])
async def test_other_clients_still_require_the_bearer(api, host):
    async with api(host) as client:
        denied = await client.get("/v1/models")
        wrong = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        allowed = await client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert denied.status_code == wrong.status_code == 401
    assert denied.json()["error"]["code"] == "invalid_api_key"
    assert allowed.status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1", "100.64.0.7", "fd7a:115c:a1e0::7"])
async def test_empty_allowlist_disables_network_authentication(api, host):
    async with api(host, {"EASEL_TRUSTED_CLIENT_CIDRS": ""}) as client:
        assert (await client.get("/v1/models")).status_code == 401


async def test_custom_client_allowlist_replaces_defaults(api):
    env = {"EASEL_TRUSTED_CLIENT_CIDRS": "192.168.0.1/32"}
    async with api("192.168.0.1", env) as client:
        assert (await client.get("/v1/models")).status_code == 200
    async with api("192.168.0.2", env) as client:
        assert (await client.get("/v1/models")).status_code == 401
    async with api("127.0.0.1", env) as client:
        assert (await client.get("/v1/models")).status_code == 401


@pytest.mark.parametrize("host", ["203.0.113.10", "10.99.0.10"])
@pytest.mark.parametrize("headers", [
    {"X-Forwarded-For": "127.0.0.1"},
    {"X-Forwarded-For": "100.64.0.7", "X-Forwarded-Proto": "https"},
    {"Forwarded": "for=127.0.0.1", "X-Real-IP": "100.64.0.7", "Host": "localhost"},
])
async def test_untrusted_peers_cannot_spoof_local_or_headscale_identity(api, host, headers):
    async with api(host, {"EASEL_TRUSTED_PROXY_CIDRS": "10.99.0.8/32"}) as client:
        assert (await client.get("/v1/models", headers=headers)).status_code == 401


@pytest.mark.parametrize("forwarded,expected", [
    ("100.64.0.7", 200),
    ("fd7a:115c:a1e0::7", 200),
    ("::ffff:100.64.0.7", 200),
    ("127.0.0.1", 200),
    ("100.64.0.7, 10.99.0.9", 200),
    ("203.0.113.10", 401),
    ("100.64.0.7, 203.0.113.10", 401),
    ("100.64.0.7, 203.0.113.10, 10.99.0.9", 401),
    ("10.99.0.9", 401),
    ("", 401),
    ("unknown", 401),
    ("100.64.0.7, unknown", 401),
    ("100.64.0.7,", 401),
    ("100.64.0.7:8080", 401),
    (None, 401),
])
async def test_proxy_chain_uses_the_first_non_proxy_from_the_right(api, forwarded, expected):
    env = {"EASEL_TRUSTED_PROXY_CIDRS": "10.99.0.8/32,10.99.0.9/32"}
    headers = {} if forwarded is None else {"X-Forwarded-For": forwarded}
    async with api("10.99.0.8", env) as client:
        assert (await client.get("/v1/models", headers=headers)).status_code == expected


async def test_duplicate_forwarded_headers_do_not_hide_an_untrusted_hop(api):
    headers = [("X-Forwarded-For", "100.64.0.7"), ("X-Forwarded-For", "203.0.113.10")]
    async with api("10.99.0.8", {"EASEL_TRUSTED_PROXY_CIDRS": "10.99.0.8/32"}) as client:
        assert (await client.get("/v1/models", headers=headers)).status_code == 401


@pytest.mark.parametrize("peer,forwarded", [
    ("100.64.0.11", None),
    ("fd7a:115c:a1e0::b", None),
    ("fd7a:115c:a1e0::b", "127.0.0.1"),
    ("::ffff:100.64.0.11", "127.0.0.1"),
    ("10.99.0.8", "100.64.0.11"),
    ("10.99.0.8", "fd7a:115c:a1e0::b"),
    ("10.99.0.8", "127.0.0.1, fd7a:115c:a1e0::b"),
    ("10.99.0.8", "127.0.0.1, 100.64.0.11"),
    ("10.99.0.8", "100.64.0.12, 100.64.0.11, 10.99.0.9"),
])
@pytest.mark.parametrize("proxy_cidrs", ["10.99.0.8/32,10.99.0.9/32",
                                         "10.99.0.8/32,10.99.0.9/32,100.64.0.11/32,fd7a:115c:a1e0::b/128"])
async def test_public_tailnet_proxy_always_requires_bearer(api, peer, forwarded, proxy_cidrs):
    env = {"EASEL_TRUSTED_PROXY_CIDRS": proxy_cidrs,
           "EASEL_REQUIRE_AUTH_CIDRS": "100.64.0.11/32,fd7a:115c:a1e0::b/128"}
    headers = {} if forwarded is None else {"X-Forwarded-For": forwarded}
    async with api(peer, env) as client:
        assert (await client.get("/v1/models", headers=headers)).status_code == 401
        headers["Authorization"] = "Bearer wrong"
        assert (await client.get("/v1/models", headers=headers)).status_code == 401
        headers["Authorization"] = "Bearer secret"
        assert (await client.get("/v1/models", headers=headers)).status_code == 200


async def test_public_proxy_exception_preserves_other_tailnet_clients(api):
    env = {"EASEL_TRUSTED_PROXY_CIDRS": "10.99.0.8/32",
           "EASEL_REQUIRE_AUTH_CIDRS": "100.64.0.11/32"}
    async with api("10.99.0.8", env) as client:
        headers = {"X-Forwarded-For": "100.64.0.12"}
        assert (await client.get("/v1/models", headers=headers)).status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1", "100.64.0.7"])
async def test_configured_proxy_cannot_inherit_its_own_client_trust(api, host):
    env = {"EASEL_TRUSTED_PROXY_CIDRS": host + "/32"}
    async with api(host, env) as client:
        assert (await client.get("/v1/models")).status_code == 401
        headers = {"X-Forwarded-For": "203.0.113.10", "Authorization": "Bearer secret"}
        assert (await client.get("/v1/models", headers=headers)).status_code == 200


@pytest.mark.parametrize("method,path,body,expected", [
    ("GET", "/v1/models", None, 200),
    ("POST", "/v1/images/generations", {"prompt": "a fox"}, 200),
    ("GET", "/v1/images/view?filename=output.png", None, 200),
    ("GET", "/v1/images/jobs/missing", None, 404),
    ("GET", "/v1/images/jobs/missing/content/0", None, 404),
    ("GET", "/v1/videos/loras", None, 200),
    ("GET", "/v1/videos/capabilities", None, 200),
    ("GET", "/v1/audio/generations/status", None, 503),
])
async def test_network_authentication_applies_to_all_api_families(api, method, path, body, expected):
    async with api("100.64.0.7") as client:
        response = await client.request(method, path, json=body)
    assert response.status_code == expected


async def test_unset_api_key_still_allows_everyone(api):
    async with api("203.0.113.10", {"EASEL_API_KEY": ""}) as client:
        assert (await client.get("/v1/models")).status_code == 200
