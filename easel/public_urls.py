"""Public media URLs without trusting caller-supplied proxy headers."""
from fastapi import Request

from .auth import trusted_client


def request_base_url(request: Request) -> str:
    settings = request.app.state.settings
    if settings.public_url and not trusted_client(settings, request):
        return settings.public_url.rstrip("/")
    return str(request.base_url).rstrip("/")
