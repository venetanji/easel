"""Authenticated audio generation backed by Suno and native ComfyUI YuE2."""
from __future__ import annotations

import asyncio
import re
import uuid
from typing import Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask

from .errors import APIError
from .api_docs import ROUTE_DOCS
from .public_urls import request_base_url
from .audio_jobs import availability, JOB_TTL
from .comfy_client import ComfyError
from . import yue2_graph

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
    suno_model: str = Field(
        default="", description="Exact native ID from GET /v1/audio/models, including custom models. "
        "Omitted or blank uses the Suno backend's configured default.",
    )
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


class Yue2Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["yue2"]
    style: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(min_length=1, max_length=12000)
    max_duration_seconds: float = Field(default=yue2_graph.DEFAULT_DURATION, strict=True,
                                        ge=0.04, le=yue2_graph.MAX_DURATION, allow_inf_nan=False)
    seed: int = Field(default=0, strict=True, ge=0, le=yue2_graph.MAX_SEED,
                      description="Exact JSON-safe integer seed, including zero; maximum 2^53-1.")

    @model_validator(mode="after")
    def validate_text(self):
        for name in ("style", "lyrics"):
            value = getattr(self, name)
            if not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must not be blank or contain NUL")
        return self


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


def _comfy_audio(request: Request):
    if not request.app.state.settings.comfy_audio_url or request.app.state.comfy_audio is None:
        raise APIError(503, "native audio backend is not configured", type="api_error",
                       code="audio_server_unavailable")
    return request.app.state.comfy_audio


def _audio_content(content: bytes, output: dict, job_id: str, request: Request, download: bool):
    etag = '"' + output["sha256"] + '"'
    headers = {"ETag": etag, "Accept-Ranges": "bytes", "Cache-Control": "private, max-age=0",
               "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{job_id}.flac"'}
    matches = [item.strip().removeprefix("W/") for item in request.headers.get("if-none-match", "").split(",")]
    if etag in matches or "*" in matches:
        return Response(status_code=304, headers=headers)
    selected = request.headers.get("range")
    if selected and request.headers.get("if-range", etag) == etag:
        match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", selected)
        if not match or not any(match.groups()):
            return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{len(content)}"})
        start = int(match[1]) if match[1] else max(0, len(content) - int(match[2]))
        end = min(len(content) - 1, int(match[2])) if match[1] and match[2] else len(content) - 1
        if start > end or start >= len(content):
            return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{len(content)}"})
        headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
        return Response(content[start:end + 1], status_code=206, media_type="audio/flac", headers=headers)
    return Response(content, media_type="audio/flac", headers=headers)


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


async def _generate_yue2(request: Request, body: Yue2Generation) -> dict:
    comfy = _comfy_audio(request)
    try:
        ready = await availability(comfy)
        if not ready["available"]:
            raise APIError(503, "YuE2 nodes, ports or checkpoint are unavailable", type="api_error",
                           code="audio_model_unavailable", details={"availability": ready})
        graph = yue2_graph.build_text_to_music(
            body.style, body.lyrics, max_duration=body.max_duration_seconds, seed=body.seed,
            filename_prefix="easel/audio/yue2/" + uuid.uuid4().hex)
        return await request.app.state.audio_jobs.submit(
            graph, seed=body.seed, max_duration=body.max_duration_seconds,
            base_url=request_base_url(request), queue_lock=request.app.state.queue_locks["audio"])
    except (httpx.HTTPError, ComfyError) as exc:
        raise APIError(503, "native audio backend is unavailable", type="api_error",
                       code="audio_server_unavailable") from exc


def _track_urls(data: dict, request: Request) -> dict:
    result = {key: value for key, value in data.items() if key not in {"path", "stream_url", "audio_url"}}
    song_id = data.get("id") or data.get("song_id")
    if song_id:
        try:
            song_id = str(UUID(str(song_id)))
        except ValueError as exc:
            raise APIError(502, "Suno returned an invalid track ID", type="api_error",
                           code="upstream_invalid_response") from exc
        base = request_base_url(request) + f"/v1/audio/tracks/{song_id}"
        result.update(audio_url=base + "/content", download_url=base + "/download")
    return result


def _generation(data: dict, request: Request) -> dict:
    result = {**data, "object": "audio.generation",
              "status_url": request_base_url(request) + "/v1/audio/generations/status"}
    if isinstance(data.get("songs"), list):
        result["songs"] = [_track_urls(song, request) for song in data["songs"]]
    return result


def register_audio_routes(app, require_auth, auth_docs=()) -> None:
    def authenticate(request: Request):
        require_auth(request.app.state.settings, request)

    router = APIRouter(prefix="/v1/audio", dependencies=[*auth_docs, Depends(authenticate)])

    @router.get("/models", **ROUTE_DOCS[("GET", "/v1/audio/models")])
    async def models(request: Request):
        data, _status = await _while_connected(request, _suno(request).request("GET", "models"))
        return data

    @router.post("/generations", **ROUTE_DOCS[("POST", "/v1/audio/generations")])
    async def generate(request: Request, body: MusicGeneration | SpeechGeneration | SoundGeneration | Yue2Generation):
        if isinstance(body, Yue2Generation):
            result = await _while_connected(request, _generate_yue2(request, body))
            return JSONResponse(result, status_code=202, headers={
                "Location": result["status_url"], "Retry-After": "2", "Cache-Control": "private, no-store"})
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

    @router.get("/capabilities", **ROUTE_DOCS[("GET", "/v1/audio/capabilities")])
    async def capabilities(request: Request, model: Literal["yue2"] = "yue2"):
        ready = {"available": False, "reason": "audio_server_unavailable"}
        if request.app.state.settings.comfy_audio_url and request.app.state.comfy_audio is not None:
            try:
                ready = await availability(request.app.state.comfy_audio)
            except (httpx.HTTPError, ComfyError):
                pass
        return {"model": model, "object": "audio.capabilities", **ready,
                "configured": bool(request.app.state.settings.comfy_audio_url),
                "planning": ["full"], "max_duration_seconds": {"default": yue2_graph.DEFAULT_DURATION,
                    "minimum": 0.04, "maximum": yue2_graph.MAX_DURATION, "is_ceiling": True},
                "seed": {"default": 0, "minimum": 0, "maximum": yue2_graph.MAX_SEED},
                "text_limits": {"style_characters": 1000, "lyrics_characters": 12000},
                "formats": ["flac"], "job_ttl_seconds": JOB_TTL,
                "guide_url": request_base_url(request) + "/v1/models/yue2/guide?mode=music",
                "license": {"weights": "CC-BY-NC-4.0", "company_commercial_use": "separate permission required"}}

    @router.get("/jobs/{job_id}", **ROUTE_DOCS[("GET", "/v1/audio/jobs/{job_id}")])
    async def audio_job(request: Request, job_id: str):
        result = await request.app.state.audio_jobs.retrieve(job_id, request_base_url(request))
        return JSONResponse(result, headers={"Cache-Control": "private, no-store"})

    @router.get("/jobs/{job_id}/content", **ROUTE_DOCS[("GET", "/v1/audio/jobs/{job_id}/content")])
    async def job_content(request: Request, job_id: str, download: bool = False):
        jobs = request.app.state.audio_jobs
        result = await jobs.retrieve(job_id, request_base_url(request))
        if result["status"] != "completed":
            raise APIError(409, "audio output is not ready", code="audio_output_not_ready")
        return _audio_content(jobs.store.content(job_id), result["output"], job_id, request, download)

    @router.get("/generations/status", **ROUTE_DOCS[("GET", "/v1/audio/generations/status")])
    async def generation_status(request: Request):
        data, _status = await _suno(request).request("GET", "generation/status")
        return _generation(data, request)

    @router.post("/generations/abandon", **ROUTE_DOCS[("POST", "/v1/audio/generations/abandon")])
    async def abandon(request: Request, body: AbandonGeneration):
        data, status = await _while_connected(request, _suno(request).request(
            "POST", "generation/abandon", json=body.model_dump(mode="json"),
        ))
        return JSONResponse(_generation(data, request), status_code=status)

    @router.get("/tracks/{song_id}", **ROUTE_DOCS[("GET", "/v1/audio/tracks/{song_id}")])
    async def track(request: Request, song_id: UUID):
        data, _status = await _while_connected(request, _suno(request).request("GET", f"songs/{song_id}"))
        return _track_urls(data, request)

    @router.post("/tracks/{song_id}/download", **ROUTE_DOCS[("POST", "/v1/audio/tracks/{song_id}/download")])
    async def download_track(request: Request, song_id: UUID):
        data, status = await _while_connected(request, _suno(request).request("POST", f"songs/{song_id}/download"))
        return JSONResponse(_track_urls(data, request), status_code=status)

    @router.get("/tracks/{song_id}/content", **ROUTE_DOCS[("GET", "/v1/audio/tracks/{song_id}/content")])
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
