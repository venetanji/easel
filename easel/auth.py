"""Bearer authentication with explicit socket-peer and reverse-proxy trust."""
from __future__ import annotations

from hmac import compare_digest
from ipaddress import IPv4Address, IPv6Address, ip_address

from fastapi import Request

from .config import Network, Settings
from .errors import APIError


def _address(value: str) -> IPv4Address | IPv6Address:
    address = ip_address(value)
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _in_networks(address: IPv4Address | IPv6Address, networks: tuple[Network, ...]) -> bool:
    return any(address in network for network in networks)


def trusted_client(settings: Settings, request: Request) -> bool:
    if request.client is None:
        return False
    try:
        address = _address(request.client.host)
        if _in_networks(address, settings.require_auth_networks):
            return False
        if _in_networks(address, settings.trusted_proxy_networks):
            forwarded = request.headers.getlist("x-forwarded-for")
            if not forwarded:
                return False
            for hop in reversed(",".join(forwarded).split(",")):
                address = _address(hop.strip())
                if _in_networks(address, settings.require_auth_networks):
                    return False
                if not _in_networks(address, settings.trusted_proxy_networks):
                    break
            else:
                return False
        return _in_networks(address, settings.trusted_client_networks)
    except ValueError:
        return False


def require_auth(settings: Settings, request: Request) -> None:
    if not settings.api_key or trusted_client(settings, request):
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if not compare_digest(token.encode(), settings.api_key.encode()):
        raise APIError(401, "invalid or missing API key", code="invalid_api_key")
