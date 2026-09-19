"""OpenAI-style error envelope."""
from __future__ import annotations

from fastapi.responses import JSONResponse


class APIError(Exception):
    """An error that maps to an OpenAI-style {"error": {...}} envelope."""

    def __init__(self, status: int, message: str, *, type: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None, retry_after: int | None = None):
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.retry_after = retry_after
        super().__init__(message)


def error_body(message: str, type: str, code: str | None = None, param: str | None = None) -> dict:
    return {"error": {"message": message, "type": type, "code": code, "param": param}}


def error_response(exc: APIError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
    return JSONResponse(
        status_code=exc.status,
        content=error_body(exc.message, exc.type, exc.code, exc.param),
        headers=headers,
    )
