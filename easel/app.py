"""FastAPI app: OpenAI-compatible image and video endpoints backed by ComfyUI."""
from __future__ import annotations

import asyncio
import base64
import math
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import File, FastAPI, Form, Request, Response, UploadFile as FastAPIUploadFile
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse

from .comfy_client import (
    ComfyClient,
    ComfyError,
    ComfyExecError,
    ComfySubmitError,
    ComfyTimeout,
    view_query,
)
from .config import MODELS, ModelSpec, Settings, UnknownModelError, parse_size, resolve_model
from .errors import APIError, error_response
from .image_jobs import ImageJobs, wants_async
from .qwen_graph import MAX_REFERENCES, reference_edit as qwen_reference_edit
from .qwen_graph import text_to_image as qwen_text_to_image
from .video_graph import VIDEO_MODEL_ID, parse_video_size, text_to_video
from .video_jobs import VIDEO_TTL_SECONDS, make_video_id, parse_video_id
from .video_controls import (guiding_nodes_available, parse_guiding_frames,
                             validate_video_form, validate_video_uploads, video_capabilities)
from .video_loras import VIDEO_LORAS, VideoLoRAError, installed_lora_names, parse_video_loras

MAX_N = 4
MAX_FANOUT = 16  # separator fan-out ceiling (one job, one GPU slot)
DEFAULT_SIZE = (1024, 1024)
DEFAULT_MODEL = "qwen-image-2.1"
MAX_QUEUED_VIDEO_JOBS = 8
VIDEO_RUNTIME_SAMPLE_LIMIT = 20


class Inflight:
    """Single-event-loop concurrency counter (check+increment is atomic without awaits)."""

    def __init__(self, max_inflight: int):
        self.max = max(1, max_inflight)
        self.n = 0

    def acquire(self) -> bool:
        if self.n >= self.max:
            return False
        self.n += 1
        return True

    def release(self) -> None:
        if self.n > 0:
            self.n -= 1


# ---- request-parsing helpers ----

def require_auth(settings: Settings, request: Request) -> None:
    if not settings.api_key:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if token != settings.api_key:
        raise APIError(401, "invalid or missing API key", code="invalid_api_key")


def resolve_spec(model: str | None):
    model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        return resolve_model(model)
    except UnknownModelError:
        raise APIError(400, f"model '{model}' not found", code="model_not_found", param="model")


def parse_n(value) -> int:
    if value in (None, ""):
        return 1
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise APIError(400, "n must be an integer", param="n")
    if not 1 <= n <= MAX_N:
        raise APIError(400, f"n must be between 1 and {MAX_N}", param="n")
    return n


def parse_size_or_400(value):
    try:
        return parse_size(value)
    except ValueError:
        raise APIError(400, f"invalid size '{value}'", param="size")


def resolve_response_format(value, settings: Settings) -> str:
    rf = (value or settings.default_response_format)
    if rf not in ("b64_json", "url"):
        raise APIError(400, "response_format must be 'b64_json' or 'url'", param="response_format")
    return rf


def resolve_server(value, settings: Settings) -> str:
    server = value or "image"
    if server not in ("image", "video"):
        raise APIError(400, "server must be 'image' or 'video'", param="server")
    if server == "video" and not settings.comfy_video_url:
        raise APIError(400, "video server is not configured", param="server")
    return server


def plan_prompts(text: str, separator: str, n: int):
    """Split a prompt on the configured separator. >1 part -> fan-out (one image
    per part, batch_size 1); else the single prompt with batch_size n."""
    parts = [p.strip() for p in text.split(separator)] if separator else [text]
    parts = [p for p in parts if p]
    if len(parts) > MAX_FANOUT:
        raise APIError(400, f"too many prompts: {len(parts)} exceeds the fan-out limit "
                            f"of {MAX_FANOUT}", param="prompt")
    if len(parts) > 1:
        return parts, 1
    return (parts[0] if parts else text), n


def int_opt(value, default):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_steps(value, spec: ModelSpec, settings: Settings) -> int:
    default = spec.default_steps if spec.default_steps is not None else settings.default_steps
    return int_opt(value, default)


def upload_name(filename: str | None) -> str:
    ext = ".png"
    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()
    return f"easel_{uuid.uuid4().hex}{ext}"


def form_str(form, key) -> str | None:
    v = form.get(key)
    if isinstance(v, str):
        return v.strip() or None
    return None


def form_images(form) -> list[UploadFile]:
    return [v for k, v in form.multi_items()
            if k in ("image", "image[]") and isinstance(v, UploadFile)]


def has_file(form, key) -> bool:
    return any(k == key and isinstance(v, UploadFile) for k, v in form.multi_items())


# ---- job execution ----

async def run_job(request: Request, graph: dict, response_format: str,
                  server: str = "image") -> list[dict]:
    settings: Settings = request.app.state.settings
    inflight: Inflight = (request.app.state.inflight_video if server == "video"
                          else request.app.state.inflight)
    comfy = request.app.state.comfy_video if server == "video" else request.app.state.comfy
    if not inflight.acquire():
        raise APIError(429, "easel is busy; retry shortly", type="rate_limit_error",
                       code="rate_limit_exceeded", retry_after=5)
    try:
        async with request.app.state.queue_locks[server]:
            await ensure_queue_capacity(comfy)
            prompt_id = await comfy.submit(graph)
        refs = await comfy.wait(prompt_id, timeout=settings.job_timeout)
    except ComfySubmitError as e:
        raise APIError(502, str(e), type="api_error", code="upstream_invalid_graph")
    except ComfyExecError as e:
        raise APIError(502, str(e), type="api_error", code="upstream_execution_error")
    except ComfyTimeout as e:
        raise APIError(504, str(e), type="api_error", code="upstream_timeout")
    except (ComfyError, httpx.HTTPError) as e:
        raise APIError(502, str(e), type="api_error", code="upstream_error")
    finally:
        inflight.release()

    if not refs:
        raise APIError(502, "comfy produced no images", type="api_error", code="upstream_no_output")

    data = []
    if response_format == "url":
        base = str(request.base_url).rstrip("/")
        for ref in refs:
            suffix = "&server=video" if server == "video" else ""
            data.append({"url": f"{base}/v1/images/view?{view_query(ref)}{suffix}"})
    else:
        for ref in refs:
            raw = await comfy.fetch(ref)
            data.append({"b64_json": base64.b64encode(raw).decode()})
    return data


async def upload_refs(comfy, images: list[UploadFile]) -> list[str]:
    names = []
    for up in images:
        data = await up.read()
        name, subfolder = await comfy.upload_image(data, upload_name(up.filename))
        names.append(f"{subfolder}/{name}" if subfolder else name)
    return names


def images_response(data: list[dict]) -> dict:
    return {"created": int(time.time()), "data": data}


async def ensure_queue_capacity(comfy, *, video=False):
    queue = await comfy.queue()
    pending = len(queue.get("queue_pending", [])) + len(queue.get("queue_running", []))
    if pending >= MAX_QUEUED_VIDEO_JOBS:
        raise APIError(429, "backend queue is full; retry shortly", type="rate_limit_error",
                       code="rate_limit_exceeded" if video else "queue_full",
                       retry_after=10 if video else 5)


@asynccontextmanager
async def image_admission(request: Request, server: str, response_format: str, model: str):
    jobs = request.app.state.image_jobs
    job_id = None
    try:
        if wants_async(request):
            async with request.app.state.queue_locks[server]:
                job_id = await jobs.admit(server, model.strip() or DEFAULT_MODEL, response_format)
                yield job_id
        else:
            yield None
    except httpx.HTTPError as exc:
        raise APIError(502, str(exc), type="api_error", code="upstream_error")
    except (sqlite3.Error, OSError):
        raise APIError(503, "image job storage is unavailable", type="api_error",
                       code="job_storage_unavailable")
    finally:
        if job_id is not None:
            with suppress(sqlite3.Error):
                jobs.store.discard_reservation(job_id)


async def image_request_response(request: Request, graph: dict, response_format: str,
                                 server: str, job_id: str | None):
    if job_id is None:
        return images_response(await run_job(request, graph, response_format, server))
    try:
        receipt = await request.app.state.image_jobs.submit(job_id, graph)
    except ComfySubmitError as exc:
        raise APIError(502, str(exc), type="api_error", code="upstream_invalid_graph")
    except ComfyError as exc:
        raise APIError(502, str(exc), type="api_error", code="upstream_error")
    return JSONResponse(status_code=202, content=receipt, headers={
        "Preference-Applied": "respond-async", "Location": f"/v1/images/jobs/{job_id}",
        "Retry-After": "2", "Cache-Control": "private, no-store",
    })


def comfy_event_time(status_info: dict, event_name: str) -> int | None:
    for message in status_info.get("messages", []):
        if isinstance(message, list) and len(message) == 2 and message[0] == event_name:
            data = message[1] if isinstance(message[1], dict) else {}
            timestamp = data.get("timestamp")
            if timestamp is not None:
                timestamp = int(timestamp)
                return timestamp // 1000 if timestamp > 100_000_000_000 else timestamp
    return None


def remember_video_runtime(state, video_id: str, duration: int | None) -> None:
    if duration is None or duration <= 0:
        return
    samples = state.video_runtimes
    samples[video_id] = duration
    while len(samples) > VIDEO_RUNTIME_SAMPLE_LIMIT:
        del samples[next(iter(samples))]


def average_video_runtime(state) -> int | None:
    samples = state.video_runtimes.values()
    if not samples:
        return None
    return round(sum(samples) / len(samples))


async def read_video_job(comfy: ComfyClient, video_id: str,
                         queue: dict | None = None) -> tuple[dict, list[dict]] | None:
    created_at, model, prompt_id = parse_video_id(video_id)
    entry = await comfy.history_item(prompt_id)
    if entry:
        status_info = entry.get("status") or {}
        outputs = entry.get("outputs") or {}
        refs = ComfyClient._collect_videos(outputs)
        started_at = comfy_event_time(status_info, "execution_start")
        interrupted = any(isinstance(message, list) and message and
                          message[0] == "execution_interrupted"
                          for message in status_info.get("messages", []))
        if status_info.get("status_str") == "error" or interrupted:
            _node_type, message = ComfyClient._extract_error(status_info)
            status = "cancelled" if interrupted else "failed"
            error = {"code": "upstream_cancelled" if interrupted else "upstream_execution_error",
                     "message": "generation was interrupted" if interrupted else message}
            progress = 0
            completed_at = comfy_event_time(status_info, "execution_error") or int(time.time())
        elif status_info.get("completed") or status_info.get("status_str") == "success":
            status = "completed" if refs else "failed"
            error = None if refs else {
                "code": "upstream_no_output", "message": "comfy produced no video output",
            }
            progress = 100 if refs else 0
            completed_at = comfy_event_time(status_info, "execution_success") or int(time.time())
        else:
            status = "in_progress"
            error = None
            progress = 5
            completed_at = None
        return ({
            "id": video_id,
            "object": "video",
            "created_at": created_at,
            "status": status,
            "completed_at": completed_at,
            "expires_at": created_at + VIDEO_TTL_SECONDS,
            "error": error,
            "model": model,
            "progress": progress,
            "_started_at": started_at,
            "_runtime_seconds": (
                max(1, completed_at - started_at)
                if status == "completed" and completed_at and started_at else None
            ),
        }, refs)

    queue = queue if queue is not None else await comfy.queue()
    for key, status, progress in (("queue_running", "in_progress", 5),
                                  ("queue_pending", "queued", 0)):
        for item in queue.get(key, []):
            if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                return ({
                    "id": video_id,
                    "object": "video",
                    "created_at": created_at,
                    "status": status,
                    "completed_at": None,
                    "expires_at": created_at + VIDEO_TTL_SECONDS,
                    "error": None,
                    "model": model,
                    "progress": progress,
                }, [])
    return None


def build_generation_graph(spec: ModelSpec, *, prompt, width, height, steps, batch_size, seed):
    return qwen_text_to_image(
        unet_name=spec.unet, clip_name=spec.clip, vae_name=spec.vae, prompt=prompt,
        width=width, height=height, steps=steps, batch_size=batch_size, seed=seed,
    )


def build_edit_graph(spec: ModelSpec, *, image_filenames, prompt, width, height, steps,
                     batch_size, seed):
    return qwen_reference_edit(
        unet_name=spec.unet, clip_name=spec.clip, vae_name=spec.vae,
        image_filenames=image_filenames, prompt=prompt, width=width, height=height,
        steps=steps, batch_size=batch_size, seed=seed,
    )


# ---- app factory ----

def create_app(settings: Settings | None = None, comfy=None, comfy_video=None) -> FastAPI:
    settings = settings or Settings.from_env()
    shared_backend = bool(settings.comfy_video_url) and (
        settings.comfy_url.rstrip("/") == settings.comfy_video_url.rstrip("/")
        or comfy is not None and comfy is comfy_video
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = None
        if comfy is None or (settings.comfy_video_url and comfy_video is None):
            http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=180.0))
            if comfy is None:
                app.state.comfy = ComfyClient(settings.comfy_url, http)
            if settings.comfy_video_url and comfy_video is None:
                app.state.comfy_video = app.state.comfy if shared_backend else \
                    ComfyClient(settings.comfy_video_url, http)
        app.state.image_jobs.backends = {"image": app.state.comfy, "video": app.state.comfy_video}
        watcher = asyncio.create_task(app.state.image_jobs.watch())
        try:
            yield
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            if http is not None:
                await http.aclose()

    app = FastAPI(title="easel", lifespan=lifespan)
    app.state.settings = settings
    app.state.inflight = Inflight(settings.max_inflight)
    app.state.inflight_video = app.state.inflight if shared_backend else Inflight(settings.max_inflight)
    image_queue_lock = asyncio.Lock()
    app.state.queue_locks = {
        "image": image_queue_lock,
        "video": image_queue_lock if shared_backend else asyncio.Lock(),
    }
    app.state.video_runtimes = {}
    app.state.video_started_at = {}
    app.state.comfy = comfy
    app.state.comfy_video = comfy_video
    app.state.image_jobs = ImageJobs(settings.image_job_dir, {"image": comfy, "video": comfy_video})

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError):
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        location = first.get("loc", ())
        param = location[-1] if location else None
        return error_response(APIError(
            400, f"invalid value for '{param}'" if param else "invalid request", param=param,
        ))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        require_auth(settings, request)
        created = 1_700_000_000
        model_ids = [*MODELS]
        if settings.comfy_video_url:
            model_ids.append(VIDEO_MODEL_ID)
        return {"object": "list", "data": [
            {"id": m, "object": "model", "created": created, "owned_by": "easel"}
            for m in model_ids
        ]}

    @app.get("/v1/videos/loras")
    async def list_video_loras(request: Request):
        require_auth(settings, request)
        resolve_server("video", settings)
        comfy = request.app.state.comfy_video
        if comfy is None:
            raise APIError(503, "video server is not available", type="api_error",
                           code="video_server_unavailable")
        try:
            installed = installed_lora_names(await comfy.object_info("LoraLoaderModelOnly"))
        except httpx.HTTPError as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        except ValueError as exc:
            raise APIError(503, str(exc), type="api_error", code="lora_loader_unavailable")
        return {"object": "list", "data": [
            {**entry, "installed": all(asset["filename"] in installed for asset in entry["files"])}
            for entry in VIDEO_LORAS.values()
        ]}

    @app.get("/v1/videos/capabilities")
    async def get_video_capabilities(request: Request):
        require_auth(settings, request)
        resolve_server("video", settings)
        comfy = request.app.state.comfy_video
        if comfy is None:
            raise APIError(503, "video server is not available", type="api_error",
                           code="video_server_unavailable")
        try:
            available = await guiding_nodes_available(comfy)
        except (ComfyError, httpx.HTTPError) as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        return video_capabilities(guides_available=available)

    @app.post("/v1/videos")
    async def create_video(
        request: Request,
        prompt: str = Form(...),
        model: str = Form(...),
        seconds: int = Form(4),
        size: str = Form("1280x720"),
        input_reference: FastAPIUploadFile | None = File(default=None),
        camera_lora: str | None = Form(None),
        camera_lora_strength: float | None = Form(None),
        loras: str | None = Form(None),
        motion_speed: float | None = Form(None),
        seed: int | None = Form(None),
        lora_reference: FastAPIUploadFile | None = File(default=None),
        lora_reference_strength: float | None = Form(None),
        guiding_frames: str | None = Form(None),
        guiding_images: list[FastAPIUploadFile] | None = File(default=None),
    ):
        require_auth(settings, request)
        resolve_server("video", settings)
        validate_video_form(await request.form())
        prompt = prompt.strip()
        model = model.strip()
        if not prompt:
            raise APIError(400, "prompt is required", param="prompt")
        if not model:
            raise APIError(400, "model is required", param="model")
        if model != VIDEO_MODEL_ID:
            raise APIError(400, f"model '{model}' not found", code="model_not_found",
                           param="model")
        if not 1 <= seconds <= 12:
            raise APIError(400, "seconds must be between 1 and 12", param="seconds")
        try:
            width, height = parse_video_size(size)
        except ValueError as exc:
            raise APIError(400, str(exc), param="size")
        if seed is not None and not 0 <= seed <= 2 ** 64 - 2:
            raise APIError(400, "seed and seed+1 must fit unsigned 64-bit integers", param="seed")
        try:
            selections = parse_video_loras(loras, camera_lora, camera_lora_strength)
        except VideoLoRAError as exc:
            raise APIError(400, str(exc), param=exc.param)
        guide_uploads = guiding_images or []
        guides = parse_guiding_frames(guiding_frames, len(guide_uploads), seconds)
        if guides and (input_reference is not None or lora_reference is not None or
                       any(entry["id"] == "ingredients" for entry, _ in selections)):
            raise APIError(400, "guiding_frames cannot be combined with input_reference or ingredients",
                           param="guiding_frames")
        ingredients = next((selection for selection in selections
                            if selection[0]["id"] == "ingredients"), None)
        if ingredients:
            if seconds < 5:
                raise APIError(400, "ingredients needs at least 5 seconds (121 reference frames)",
                               param="seconds")
            if lora_reference is None:
                raise APIError(400, "ingredients requires a PNG, JPEG, or WebP reference sheet",
                               param="lora_reference")
            if lora_reference.content_type not in ("image/png", "image/jpeg", "image/webp"):
                raise APIError(400, "lora_reference must be a PNG, JPEG, or WebP image",
                               param="lora_reference")
            if len(selections) != 1:
                raise APIError(400, "ingredients cannot yet be stacked with other LoRAs", param="loras")
        elif lora_reference is not None or lora_reference_strength is not None:
            raise APIError(400, "lora_reference requires the ingredients LoRA", param="lora_reference")
        if lora_reference_strength is not None and (
                not math.isfinite(lora_reference_strength) or not 0 <= lora_reference_strength <= 1):
            raise APIError(400, "lora_reference_strength must be between 0 and 1",
                           param="lora_reference_strength")
        has_slow_motion = any(entry["id"] == "slow-motion" for entry, _strength in selections)
        if has_slow_motion and motion_speed is None:
            raise APIError(400, "slow-motion requires motion_speed", param="motion_speed")
        if motion_speed is not None:
            if not has_slow_motion:
                raise APIError(400, "motion_speed requires the slow-motion LoRA", param="motion_speed")
            if not math.isfinite(motion_speed) or not 0.025 <= motion_speed <= 1:
                raise APIError(400, "motion_speed must be between 0.025 and 1", param="motion_speed")
        for entry, _strength in selections:
            if "input_reference" in entry["requires"] and input_reference is None:
                raise APIError(400, f"{entry['id']} requires input_reference", param="input_reference")
        if any(entry["id"] == "cinemagraph" for entry, _strength in selections) and \
                any(entry["kind"] == "camera" and entry["id"] != "camera-static"
                    for entry, _strength in selections):
            raise APIError(400, "cinemagraph requires a static camera; moving camera LoRAs conflict",
                           param="loras")

        uploads = [(name, upload) for name, upload in (
            ("input_reference", input_reference), ("lora_reference", lora_reference)) if upload is not None]
        uploads.extend(("guiding_images", upload) for upload in guide_uploads)
        await validate_video_uploads(uploads)

        comfy = request.app.state.comfy_video
        if comfy is None:
            raise APIError(503, "video server is not available", type="api_error",
                           code="video_server_unavailable")
        if guides:
            try:
                available = await guiding_nodes_available(comfy)
            except (ComfyError, httpx.HTTPError) as exc:
                raise APIError(502, str(exc), type="api_error", code="upstream_error")
            if not available:
                raise APIError(503, "ComfyUI lacks compatible guiding-frame nodes", type="api_error",
                               code="guiding_nodes_unavailable", param="guiding_frames")
        if selections:
            try:
                installed = installed_lora_names(await comfy.object_info("LoraLoaderModelOnly"))
            except httpx.HTTPError as exc:
                raise APIError(502, str(exc), type="api_error", code="upstream_error")
            except ValueError as exc:
                raise APIError(503, str(exc), type="api_error", code="lora_loader_unavailable")
            missing = [entry["id"] for entry, _strength in selections
                       if entry["files"][0]["filename"] not in installed]
            if missing:
                raise APIError(503, "LoRA weights are not installed: " + ", ".join(missing),
                               type="api_error", code="lora_not_installed", param="loras")
        if ingredients:
            for node_name in ("LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide", "LTXVCropGuides"):
                try:
                    info = await comfy.object_info(node_name)
                except httpx.HTTPError as exc:
                    raise APIError(502, str(exc), type="api_error", code="upstream_error")
                if node_name not in info:
                    raise APIError(503, f"ComfyUI is missing {node_name}", type="api_error",
                                   code="ic_lora_nodes_unavailable")
        async with request.app.state.queue_locks["video"]:
            try:
                await ensure_queue_capacity(comfy, video=True)
            except httpx.HTTPError as exc:
                raise APIError(502, str(exc), type="api_error", code="upstream_error")

            image_name = None
            if input_reference is not None:
                content_type = input_reference.content_type or "application/octet-stream"
                if content_type not in ("image/png", "image/jpeg", "image/webp"):
                    raise APIError(400, "input_reference must be a PNG, JPEG, or WebP image",
                                   param="input_reference")
                try:
                    name, subfolder = await comfy.upload_image(
                        await input_reference.read(),
                        upload_name(input_reference.filename),
                        content_type,
                    )
                except (ComfyError, httpx.HTTPError) as exc:
                    raise APIError(502, str(exc), type="api_error", code="upstream_upload_error")
                image_name = f"{subfolder}/{name}" if subfolder else name

            created_at = int(time.time())
            reference_name = None
            if lora_reference is not None:
                content_type = lora_reference.content_type or "application/octet-stream"
                if content_type not in ("image/png", "image/jpeg", "image/webp"):
                    raise APIError(400, "lora_reference must be a PNG, JPEG, or WebP image",
                                   param="lora_reference")
                try:
                    name, subfolder = await comfy.upload_image(await lora_reference.read(),
                        upload_name(lora_reference.filename), content_type)
                except (ComfyError, httpx.HTTPError) as exc:
                    raise APIError(502, str(exc), type="api_error", code="upstream_upload_error")
                reference_name = f"{subfolder}/{name}" if subfolder else name
            guide_names = []
            for upload in guide_uploads:
                try:
                    name, subfolder = await comfy.upload_image(await upload.read(),
                        upload_name(upload.filename), upload.content_type)
                except (ComfyError, httpx.HTTPError) as exc:
                    raise APIError(502, str(exc), type="api_error", code="upstream_upload_error")
                guide_names.append(f"{subfolder}/{name}" if subfolder else name)
            prefix = f"easel/videos/{uuid.uuid4().hex}"
            graph = text_to_video(
                prompt=", ".join([entry["trigger"] for entry, _strength in selections
                                  if entry["trigger"] and entry["trigger"] not in prompt] + [prompt]),
                seconds=seconds,
                width=width,
                height=height,
                filename_prefix=prefix,
                input_image=image_name,
                seed=seed,
                loras=[(entry["files"][0]["filename"], strength) for entry, strength in selections
                       if entry["kind"] != "ic"],
                motion_speed=motion_speed,
                ic_lora=(ingredients[0]["files"][0]["filename"], ingredients[1]) if ingredients else None,
                reference_image=reference_name,
                reference_strength=lora_reference_strength if lora_reference_strength is not None else 1.0,
                guiding_frames=[(guide_names[guide.image_index], guide.frame_index, guide.strength)
                                for guide in guides],
            )
            try:
                prompt_id = await comfy.submit(graph)
            except ComfySubmitError as exc:
                raise APIError(502, str(exc), type="api_error", code="upstream_invalid_graph")
            except (ComfyError, httpx.HTTPError) as exc:
                raise APIError(502, str(exc), type="api_error", code="upstream_error")

        video_id = make_video_id(prompt_id, model, created_at)
        return {
            "id": video_id,
            "object": "video",
            "created_at": created_at,
            "status": "queued",
            "completed_at": None,
            "expires_at": created_at + VIDEO_TTL_SECONDS,
            "error": None,
            "model": model,
            "progress": 0,
        }

    @app.get("/v1/videos/{video_id}")
    async def retrieve_video(request: Request, video_id: str):
        require_auth(settings, request)
        comfy = request.app.state.comfy_video
        if comfy is None:
            raise APIError(503, "video server is not available", type="api_error",
                           code="video_server_unavailable")
        try:
            result = await read_video_job(comfy, video_id)
        except ValueError as exc:
            raise APIError(404, str(exc), code="video_not_found")
        except httpx.HTTPError as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        if result is None:
            raise APIError(404, f"video '{video_id}' not found", code="video_not_found")
        video = dict(result[0])
        remember_video_runtime(request.app.state, video_id, video.pop("_runtime_seconds", None))
        if video["status"] == "in_progress":
            request.app.state.video_started_at.setdefault(video_id, int(time.time()))
        else:
            request.app.state.video_started_at.pop(video_id, None)
        video.pop("_started_at", None)
        return video

    @app.get("/v1/videos/queue/{video_id}")
    async def video_queue_item(request: Request, video_id: str):
        require_auth(settings, request)
        comfy = request.app.state.comfy_video
        if comfy is None:
            raise APIError(503, "video server is not available", type="api_error",
                           code="video_server_unavailable")
        try:
            queue = await comfy.queue()
            result = await read_video_job(comfy, video_id, queue=queue)
        except ValueError as exc:
            raise APIError(404, str(exc), code="video_not_found")
        except httpx.HTTPError as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        if result is None:
            raise APIError(404, f"video '{video_id}' not found", code="video_not_found")

        video, _refs = result
        remember_video_runtime(request.app.state, video_id, video.get("_runtime_seconds"))
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])
        created_at, _model, prompt_id = parse_video_id(video_id)
        runtime = average_video_runtime(request.app.state)
        queue_position = None
        queue_ahead = 0
        estimated_wait = None

        running_index = next((i for i, item in enumerate(running)
                              if isinstance(item, list) and len(item) > 1
                              and item[1] == prompt_id), None)
        pending_index = next((i for i, item in enumerate(pending)
                              if isinstance(item, list) and len(item) > 1
                              and item[1] == prompt_id), None)
        if video["status"] == "completed":
            estimated_wait = 0
        elif video["status"] not in ("failed", "cancelled") and running_index is not None:
            queue_position = 0
            started_at = video.get("_started_at") or request.app.state.video_started_at.setdefault(
                video_id, int(time.time()))
            if runtime is not None:
                elapsed = max(0, int(time.time()) - started_at) if started_at else 0
                estimated_wait = max(0, runtime - elapsed)
        elif video["status"] not in ("failed", "cancelled") and pending_index is not None:
            queue_position = pending_index + 1
            queue_ahead = len(running) + pending_index
            if runtime is not None:
                estimated_wait = runtime * (queue_ahead + 1)

        now = int(time.time())
        if video["status"] in ("completed", "failed", "cancelled"):
            request.app.state.video_started_at.pop(video_id, None)
        return {
            "object": "video_queue_item",
            "id": video_id,
            "status": video["status"],
            "created_at": created_at,
            "expires_at": video["expires_at"],
            "completed_at": video["completed_at"],
            "progress": video["progress"],
            "error": video["error"],
            "queue_position": queue_position,
            "queue_ahead": queue_ahead,
            "running": len(running),
            "pending": len(pending),
            "average_generation_seconds": runtime,
            "estimated_wait_seconds": estimated_wait,
            "estimated_completion_at": (
                video["completed_at"] if video["status"] == "completed"
                else now + estimated_wait if estimated_wait is not None else None
            ),
        }

    @app.get("/v1/videos/{video_id}/content")
    async def video_content(request: Request, video_id: str):
        require_auth(settings, request)
        comfy = request.app.state.comfy_video
        if comfy is None:
            raise APIError(503, "video server is not available", type="api_error",
                           code="video_server_unavailable")
        try:
            result = await read_video_job(comfy, video_id)
        except ValueError as exc:
            raise APIError(404, str(exc), code="video_not_found")
        except httpx.HTTPError as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        if result is None:
            raise APIError(404, f"video '{video_id}' not found", code="video_not_found")
        video, refs = result
        if video["status"] != "completed":
            message = video["error"]["message"] if video["error"] else \
                "video generation has not completed"
            raise APIError(409, message, code="video_not_ready")
        try:
            raw = await comfy.fetch(refs[0])
        except (ComfyError, httpx.HTTPError) as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        return Response(
            content=raw,
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="{video_id}.mp4"'},
        )

    @app.post("/v1/images/generations")
    async def generations(request: Request):
        require_auth(settings, request)
        try:
            body = await request.json()
        except Exception:
            raise APIError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise APIError(400, "request body must be a JSON object")
        prompt = body.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise APIError(400, "prompt is required", param="prompt")
        spec = resolve_spec(body.get("model"))
        server = resolve_server(body.get("server"), settings)
        n = parse_n(body.get("n"))
        rf = resolve_response_format(body.get("response_format"), settings)
        size = parse_size_or_400(body.get("size"))
        width, height = size or DEFAULT_SIZE
        prompt_arg, batch = plan_prompts(prompt, settings.prompt_separator, n)
        steps = resolve_steps(body.get("steps"), spec, settings)
        graph = build_generation_graph(
            spec, prompt=prompt_arg,
            width=width, height=height,
            steps=steps,
            batch_size=batch, seed=int_opt(body.get("seed"), None),
        )
        async with image_admission(request, server, rf, body.get("model") or DEFAULT_MODEL) as job_id:
            return await image_request_response(request, graph, rf, server, job_id)

    @app.post("/v1/images/edits")
    async def edits(request: Request):
        require_auth(settings, request)
        form = await request.form()
        if has_file(form, "mask"):
            raise APIError(400, "masked inpainting is not supported; this model does "
                                "instruction/reference editing", code="unsupported_parameter", param="mask")
        images = form_images(form)
        if not images:
            raise APIError(400, "at least one image is required", param="image")
        prompt = form_str(form, "prompt")
        if not prompt:
            raise APIError(400, "prompt is required", param="prompt")
        spec = resolve_spec(form_str(form, "model"))
        server = resolve_server(form_str(form, "server"), settings)
        if spec.backend == "qwen_image_2_1" and len(images) > MAX_REFERENCES:
            raise APIError(
                400, f"Qwen Image 2.1 supports at most {MAX_REFERENCES} reference images",
                param="image",
            )
        n = parse_n(form_str(form, "n"))
        rf = resolve_response_format(form_str(form, "response_format"), settings)
        size = parse_size_or_400(form_str(form, "size"))
        width, height = size if size else (None, None)
        comfy_client = request.app.state.comfy_video if server == "video" else request.app.state.comfy
        prompt_arg, batch = plan_prompts(prompt, settings.prompt_separator, n)
        steps = resolve_steps(form_str(form, "steps"), spec, settings)
        async with image_admission(request, server, rf, form_str(form, "model") or DEFAULT_MODEL) as job_id:
            names = await upload_refs(comfy_client, images)
            graph = build_edit_graph(
                spec, image_filenames=names, prompt=prompt_arg,
                width=width, height=height,
                steps=steps,
                batch_size=batch, seed=int_opt(form_str(form, "seed"), None),
            )
            return await image_request_response(request, graph, rf, server, job_id)

    @app.post("/v1/images/variations")
    async def variations(request: Request):
        require_auth(settings, request)
        form = await request.form()
        images = form_images(form)
        if not images:
            raise APIError(400, "an image is required", param="image")
        spec = resolve_spec(form_str(form, "model"))
        server = resolve_server(form_str(form, "server"), settings)
        n = parse_n(form_str(form, "n"))
        rf = resolve_response_format(form_str(form, "response_format"), settings)
        size = parse_size_or_400(form_str(form, "size"))
        width, height = size if size else (None, None)
        comfy_client = request.app.state.comfy_video if server == "video" else request.app.state.comfy
        prompt_arg, batch = plan_prompts(settings.variation_prompt, settings.prompt_separator, n)
        steps = resolve_steps(form_str(form, "steps"), spec, settings)
        async with image_admission(request, server, rf, form_str(form, "model") or DEFAULT_MODEL) as job_id:
            names = await upload_refs(comfy_client, images[:1])
            graph = build_edit_graph(
                spec, image_filenames=names, prompt=prompt_arg, width=width, height=height,
                steps=steps,
                batch_size=batch, seed=int_opt(form_str(form, "seed"), None),
            )
            return await image_request_response(request, graph, rf, server, job_id)

    @app.get("/v1/images/jobs/{job_id}")
    async def image_job(request: Request, job_id: str):
        require_auth(settings, request)
        try:
            result = await request.app.state.image_jobs.retrieve(job_id, str(request.base_url).rstrip("/"))
            return JSONResponse(content=result, headers={"Cache-Control": "private, no-store"})
        except httpx.HTTPError as exc:
            raise APIError(502, str(exc), type="api_error", code="upstream_error")
        except (sqlite3.Error, OSError):
            raise APIError(503, "image job storage is unavailable", type="api_error",
                           code="job_storage_unavailable")

    @app.get("/v1/images/jobs/{job_id}/content/{index}")
    async def image_job_content(request: Request, job_id: str, index: int):
        require_auth(settings, request)
        store = request.app.state.image_jobs.store
        try:
            job = store.get(job_id)
            if job["status"] != "completed":
                raise APIError(409, "image job is not completed", code="image_job_not_completed")
            images = store.outputs(job_id)
        except (sqlite3.Error, OSError):
            raise APIError(503, "image job storage is unavailable", type="api_error",
                           code="job_storage_unavailable")
        if not 0 <= index < len(images):
            raise APIError(404, "image output not found", code="image_output_not_found")
        return Response(content=images[index], media_type="image/png",
                        headers={"Cache-Control": "private, no-store"})

    @app.get("/v1/images/view")
    async def view(request: Request, filename: str, subfolder: str = "", type: str = "output",
                   server: str = "image"):
        require_auth(settings, request)
        if type not in ("output", "input", "temp"):
            raise APIError(400, "invalid type", param="type")
        server = resolve_server(server, settings)
        comfy_client = request.app.state.comfy_video if server == "video" else request.app.state.comfy
        raw = await comfy_client.fetch(
            {"filename": filename, "subfolder": subfolder, "type": type})
        return Response(content=raw, media_type="image/png")

    return app


app = create_app()  # ASGI target for `uvicorn easel.app:app`
