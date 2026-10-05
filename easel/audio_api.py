"""Authenticated audio generation backed by Suno's shared browser session."""
from __future__ import annotations

import asyncio
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask

from .errors import APIError

AUDIO_MODELS = ("suno-music", "suno-speech", "suno-sound")


class AudioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dry_run: bool = False


class MusicGeneration(AudioRequest):
    model: Literal["suno-music"] = "suno-music"
    prompt: str = Field(default="", max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
    title: str = ""
    make_instrumental: bool = False
    negative_prompt: str = Field(default="", max_length=1000)
    suno_model: str = ""
    weirdness: int | None = Field(default=None, ge=0, le=100)
    style_influence: int | None = Field(default=None, ge=0, le=100)
    variety: int | None = Field(default=None, ge=0, le=4)
    vocal_gender: Literal["", "male", "female"] = ""
    duration_seconds: int | None = Field(default=None, ge=10, le=360)
    reference_audio_id: UUID | None = None
    audio_mode: Literal["cover", "extend"] = "cover"
    audio_influence: int | None = Field(default=None, ge=0, le=100)
    inspiration_ids: list[UUID] | None = Field(default=None, max_length=4)
    inspiration_playlist: str = ""

    @model_validator(mode="after")
    def validate_music(self):
        if not (self.prompt.strip() or self.lyrics.strip()):
            raise ValueError("provide a music prompt or lyrics")
        if self.reference_audio_id and (self.inspiration_ids or self.inspiration_playlist):
            raise ValueError("choose reference audio or inspiration, not both")
        if self.make_instrumental and self.lyrics.strip():
            raise ValueError("instrumental generation requires empty lyrics")
        if self.audio_influence is not None and not self.reference_audio_id:
            raise ValueError("audio_influence requires reference_audio_id")
        if self.inspiration_ids and self.inspiration_playlist:
            raise ValueError("choose inspiration IDs or a playlist, not both")
        if self.inspiration_ids and len(set(self.inspiration_ids)) != len(self.inspiration_ids):
            raise ValueError("inspiration_ids must be unique")
        return self


class SpeechGeneration(AudioRequest):
    model: Literal["suno-speech"]
    prompt: str = Field(min_length=1, max_length=5000)
    tone: str = Field(default="", max_length=1000)
    background_music: bool = False
    vocal_gender: Literal["", "male", "female"] = ""
    variety: int | None = Field(default=None, ge=0, le=4)


class SoundGeneration(AudioRequest):
    model: Literal["suno-sound"]
    prompt: str = Field(min_length=1, max_length=500)
    sound_type: Literal["one_shot", "loop"] = "one_shot"
    bpm: int | None = Field(default=None, ge=1, le=300)


class AbandonGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt_id: UUID


def _suno(request: Request):
    if not request.app.state.settings.suno_url:
        raise APIError(503, "audio backend is not configured", type="api_error",
                       code="audio_server_unavailable")
    if request.app.state.suno is None:
        raise APIError(503, "audio backend is not available", type="api_error",
                       code="audio_server_unavailable")
    return request.app.state.suno


async def _while_connected(request: Request, operation):
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    watcher = asyncio.create_task(disconnected())
    worker = asyncio.create_task(operation)
    try:
        done, _pending = await asyncio.wait((worker, watcher), return_when=asyncio.FIRST_COMPLETED)
        if worker in done:
            return await worker
        await watcher
        raise APIError(499, "Caller disconnected; check generation status before retrying",
                       type="api_error", code="client_disconnected")
    finally:
        worker.cancel()
        watcher.cancel()
        await asyncio.gather(worker, watcher, return_exceptions=True)


def _track_urls(data: dict, request: Request) -> dict:
    result = {key: value for key, value in data.items() if key not in {"path", "stream_url", "audio_url"}}
    song_id = data.get("id") or data.get("song_id")
    if song_id:
        try:
            song_id = str(UUID(str(song_id)))
        except ValueError as exc:
            raise APIError(502, "Suno returned an invalid track ID", type="api_error",
                           code="upstream_invalid_response") from exc
        base = str(request.base_url).rstrip("/") + f"/v1/audio/tracks/{song_id}"
        result.update(audio_url=base + "/content", download_url=base + "/download")
    return result


def _generation(data: dict, request: Request) -> dict:
    result = {**data, "object": "audio.generation",
              "status_url": str(request.base_url).rstrip("/") + "/v1/audio/generations/status"}
    if isinstance(data.get("songs"), list):
        result["songs"] = [_track_urls(song, request) for song in data["songs"]]
    return result


def register_audio_routes(app, require_auth) -> None:
    def authenticate(request: Request):
        require_auth(request.app.state.settings, request)

    router = APIRouter(prefix="/v1/audio", tags=["Audio"], dependencies=[Depends(authenticate)])

    @router.post("/generations")
    async def generate(request: Request, body: MusicGeneration | SpeechGeneration | SoundGeneration):
        suno = _suno(request)
        if not body.prompt.strip() and not getattr(body, "lyrics", "").strip():
            raise APIError(400, "prompt must not be blank", param="prompt")
        payload = body.model_dump(mode="json", exclude={"model"}, exclude_none=True)
        if isinstance(body, MusicGeneration):
            kind = "music"
            payload["tags"] = payload.pop("prompt")
            payload["model"] = payload.pop("suno_model")
        elif isinstance(body, SpeechGeneration):
            kind = "speech"
            payload["script"] = payload.pop("prompt")
        else:
            kind = "sound"
        data, status = await _while_connected(request, suno.request("POST", f"generate/{kind}", json=payload))
        result = {**_generation(data, request), "model": body.model}
        headers = {"Location": result["status_url"], "Retry-After": "5"} if status == 202 else None
        return JSONResponse(result, status_code=status, headers=headers)

    @router.get("/generations/status")
    async def generation_status(request: Request):
        data, _status = await _suno(request).request("GET", "generation/status")
        return _generation(data, request)

    @router.post("/generations/abandon")
    async def abandon(request: Request, body: AbandonGeneration):
        data, status = await _while_connected(request, _suno(request).request(
            "POST", "generation/abandon", json=body.model_dump(mode="json"),
        ))
        return JSONResponse(_generation(data, request), status_code=status)

    @router.get("/tracks/{song_id}")
    async def track(request: Request, song_id: UUID):
        data, _status = await _while_connected(request, _suno(request).request("GET", f"songs/{song_id}"))
        return _track_urls(data, request)

    @router.post("/tracks/{song_id}/download")
    async def download_track(request: Request, song_id: UUID):
        data, status = await _while_connected(request, _suno(request).request("POST", f"songs/{song_id}/download"))
        return JSONResponse(_track_urls(data, request), status_code=status)

    @router.get("/tracks/{song_id}/content")
    async def audio_content(request: Request, song_id: UUID, download: bool = False):
        upstream = await _suno(request).audio(
            str(song_id), download=download,
            headers={name: request.headers[name] for name in (
                "range", "if-range", "if-none-match", "if-modified-since",
            ) if name in request.headers},
        )
        headers = {name: upstream.headers[name] for name in (
            "content-type", "content-length", "content-disposition", "content-range",
            "content-encoding", "accept-ranges", "etag", "last-modified",
        ) if name in upstream.headers}
        async def chunks():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            chunks(), status_code=upstream.status_code, headers=headers,
            background=BackgroundTask(upstream.aclose),
        )

    app.include_router(router)
