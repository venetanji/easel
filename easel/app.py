"""FastAPI app: OpenAI-compatible image endpoints backed by ComfyUI Flux.2."""
from __future__ import annotations

import base64
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from starlette.datastructures import UploadFile

from .comfy_client import (
    ComfyClient,
    ComfyError,
    ComfyExecError,
    ComfySubmitError,
    ComfyTimeout,
    view_query,
)
from .config import MODELS, Settings, UnknownModelError, parse_size, resolve_model
from .errors import APIError, error_response
from .flux_graph import reference_edit, text_to_image

MAX_N = 4
DEFAULT_SIZE = (1024, 1024)
DEFAULT_MODEL = "flux2-9b"


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


def plan_prompts(text: str, separator: str, n: int):
    """Split a prompt on the configured separator. >1 part -> fan-out (one image
    per part, batch_size 1); else the single prompt with batch_size n."""
    parts = [p.strip() for p in text.split(separator)] if separator else [text]
    parts = [p for p in parts if p]
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

async def run_job(request: Request, graph: dict, response_format: str) -> list[dict]:
    settings: Settings = request.app.state.settings
    inflight: Inflight = request.app.state.inflight
    comfy = request.app.state.comfy
    if not inflight.acquire():
        raise APIError(429, "easel is busy; retry shortly", type="rate_limit_error",
                       code="rate_limit_exceeded", retry_after=5)
    try:
        prompt_id = await comfy.submit(graph)
        refs = await comfy.wait(prompt_id, timeout=settings.job_timeout)
    except ComfySubmitError as e:
        raise APIError(502, str(e), type="api_error", code="upstream_invalid_graph")
    except ComfyExecError as e:
        raise APIError(502, str(e), type="api_error", code="upstream_execution_error")
    except ComfyTimeout as e:
        raise APIError(504, str(e), type="api_error", code="upstream_timeout")
    except ComfyError as e:
        raise APIError(502, str(e), type="api_error", code="upstream_error")
    finally:
        inflight.release()

    if not refs:
        raise APIError(502, "comfy produced no images", type="api_error", code="upstream_no_output")

    data = []
    if response_format == "url":
        base = str(request.base_url).rstrip("/")
        for ref in refs:
            data.append({"url": f"{base}/v1/images/view?{view_query(ref)}"})
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


# ---- app factory ----

def create_app(settings: Settings | None = None, comfy=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if comfy is None:
            http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=180.0))
            app.state.comfy = ComfyClient(settings.comfy_url, http)
            try:
                yield
            finally:
                await http.aclose()
        else:
            yield

    app = FastAPI(title="easel", lifespan=lifespan)
    app.state.settings = settings
    app.state.inflight = Inflight(settings.max_inflight)
    app.state.comfy = comfy

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError):
        return error_response(exc)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        require_auth(settings, request)
        created = 1_700_000_000
        return {"object": "list", "data": [
            {"id": m, "object": "model", "created": created, "owned_by": "easel"} for m in MODELS
        ]}

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
        n = parse_n(body.get("n"))
        rf = resolve_response_format(body.get("response_format"), settings)
        size = parse_size_or_400(body.get("size"))
        width, height = size or DEFAULT_SIZE
        prompt_arg, batch = plan_prompts(prompt, settings.prompt_separator, n)
        graph = text_to_image(
            unet_name=spec.unet, clip_name=spec.clip, prompt=prompt_arg,
            width=width, height=height,
            steps=int_opt(body.get("steps"), settings.default_steps),
            batch_size=batch, seed=int_opt(body.get("seed"), None),
        )
        return images_response(await run_job(request, graph, rf))

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
        n = parse_n(form_str(form, "n"))
        rf = resolve_response_format(form_str(form, "response_format"), settings)
        size = parse_size_or_400(form_str(form, "size"))
        width, height = size if size else (None, None)
        names = await upload_refs(request.app.state.comfy, images)
        prompt_arg, batch = plan_prompts(prompt, settings.prompt_separator, n)
        graph = reference_edit(
            unet_name=spec.unet, clip_name=spec.clip, image_filenames=names, prompt=prompt_arg,
            width=width, height=height,
            steps=int_opt(form_str(form, "steps"), settings.default_steps),
            batch_size=batch, seed=int_opt(form_str(form, "seed"), None),
        )
        return images_response(await run_job(request, graph, rf))

    @app.post("/v1/images/variations")
    async def variations(request: Request):
        require_auth(settings, request)
        form = await request.form()
        images = form_images(form)
        if not images:
            raise APIError(400, "an image is required", param="image")
        spec = resolve_spec(form_str(form, "model"))
        n = parse_n(form_str(form, "n"))
        rf = resolve_response_format(form_str(form, "response_format"), settings)
        size = parse_size_or_400(form_str(form, "size"))
        width, height = size if size else (None, None)
        names = await upload_refs(request.app.state.comfy, images[:1])
        prompt_arg, batch = plan_prompts(settings.variation_prompt, settings.prompt_separator, n)
        graph = reference_edit(
            unet_name=spec.unet, clip_name=spec.clip, image_filenames=names,
            prompt=prompt_arg, width=width, height=height,
            steps=int_opt(form_str(form, "steps"), settings.default_steps),
            batch_size=batch, seed=int_opt(form_str(form, "seed"), None),
        )
        return images_response(await run_job(request, graph, rf))

    @app.get("/v1/images/view")
    async def view(request: Request, filename: str, subfolder: str = "", type: str = "output"):
        if type not in ("output", "input", "temp"):
            raise APIError(400, "invalid type", param="type")
        raw = await request.app.state.comfy.fetch(
            {"filename": filename, "subfolder": subfolder, "type": type})
        return Response(content=raw, media_type="image/png")

    return app


app = create_app()  # ASGI target for `uvicorn easel.app:app`
