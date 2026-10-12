"""A public proxy origin never comes from unvalidated Host/Proto headers."""
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from easel.config import Settings
from easel.public_urls import request_base_url


@pytest.mark.parametrize("host,forwarded,expected", [
    ("10.99.0.8", "100.64.0.11", "https://easel.ait4x.org"),
    ("100.64.0.11", "127.0.0.1", "https://easel.ait4x.org"),
    ("fd7a:115c:a1e0::b", "127.0.0.1", "https://easel.ait4x.org"),
    ("203.0.113.10", "127.0.0.1", "https://easel.ait4x.org"),
    ("10.99.0.8", "100.64.0.12", "http://easel.tail.ait4x.org"),
    ("127.0.0.1", "203.0.113.10", "http://easel.tail.ait4x.org"),
])
def test_public_urls_follow_the_network_authentication_boundary(host, forwarded, expected):
    settings = Settings.from_env({
        "EASEL_PUBLIC_URL": "https://easel.ait4x.org/",
        "EASEL_REQUIRE_AUTH_CIDRS": "100.64.0.11/32,fd7a:115c:a1e0::b/128",
        "EASEL_TRUSTED_PROXY_CIDRS": "10.99.0.8/32",
    })
    request = Request({"type": "http", "scheme": "http", "path": "/v1/models",
        "root_path": "", "query_string": b"", "server": ("easel.tail.ait4x.org", 80),
        "client": (host, 123), "headers": [(b"host", b"easel.tail.ait4x.org"),
            (b"x-forwarded-for", forwarded.encode()), (b"x-forwarded-host", b"attacker.example"),
            (b"x-forwarded-proto", b"https")],
        "app": SimpleNamespace(state=SimpleNamespace(settings=settings))})
    assert request_base_url(request) == expected
