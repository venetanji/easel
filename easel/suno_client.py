"""Suno REST transport. Generation POSTs are never retried."""
from __future__ import annotations

import httpx

from .errors import APIError


class SunoClient:
    def __init__(self, base_url: str, http: httpx.AsyncClient, *, token: str | None = None,
                 timeout: float = 180.0):
        self.base_url = base_url.rstrip("/") + "/api/v1/"
        self.http = http
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout

    def _upstream_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
            detail = body.get("error", {}) if isinstance(body, dict) else {}
        except ValueError:
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        status = response.status_code
        code = detail.get("code") or "upstream_error"
        message = detail.get("message") or "Suno backend request failed"
        if status in (401, 403):
            status, code, message = 502, "upstream_auth_error", "Suno backend authentication failed"
        elif status == 422:
            status = 400
        elif status not in (400, 404, 409, 413, 416, 429, 503, 504):
            status = 502
        retry_after = response.headers.get("Retry-After", "")
        retry_seconds = int(retry_after) if retry_after.isdigit() else None
        raise APIError(
            status, message, type="invalid_request_error" if status < 500 else "api_error",
            code=code, retry_after=(retry_seconds if retry_seconds is not None else 5) if status in (429, 503) else None,
            details={key: value for key, value in detail.items() if key not in {"code", "message"}},
        )

    async def request(self, method: str, path: str, **kwargs) -> tuple[dict, int]:
        try:
            response = await self.http.request(
                method, self.base_url + path, headers=self.headers, timeout=self.timeout,
                follow_redirects=False, **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise APIError(
                504, "Suno request timed out; check generation status before retrying",
                type="api_error", code="upstream_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise APIError(
                502, "Suno backend is unreachable; check generation status before retrying",
                type="api_error", code="upstream_error",
            ) from exc
        self._upstream_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise APIError(502, "Suno returned an invalid response", type="api_error",
                           code="upstream_invalid_response") from exc
        if not isinstance(data, dict) or not 200 <= response.status_code < 300:
            raise APIError(502, "Suno returned an invalid response", type="api_error",
                           code="upstream_invalid_response")
        return data, response.status_code

    async def audio(self, song_id: str, *, download: bool = False,
                    headers: dict | None = None) -> httpx.Response:
        request = self.http.build_request(
            "GET", self.base_url + f"songs/{song_id}/audio",
            headers={**(headers or {}), **self.headers}, params={"download": str(download).lower()},
            timeout=self.timeout,
        )
        try:
            response = await self.http.send(request, stream=True, follow_redirects=False)
        except httpx.TimeoutException as exc:
            raise APIError(504, "Suno audio request timed out", type="api_error",
                           code="upstream_timeout") from exc
        except httpx.HTTPError as exc:
            raise APIError(502, "Suno backend is unreachable", type="api_error",
                           code="upstream_error") from exc
        try:
            if response.status_code >= 400:
                await response.aread()
            self._upstream_error(response)
            content_type = response.headers.get("content-type", "").split(";", 1)[0]
            if response.status_code not in (200, 206, 304) or (
                response.status_code != 304 and not (
                    content_type.startswith("audio/") or content_type in {
                        "video/mp4", "video/webm", "application/ogg", "application/octet-stream",
                    }
                )
            ):
                raise APIError(502, "Suno returned an invalid audio response", type="api_error",
                               code="upstream_invalid_response")
        except BaseException:
            await response.aclose()
            raise
        return response
