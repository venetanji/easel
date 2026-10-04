"""Async ComfyUI REST client: upload refs, submit graphs, poll history, fetch images."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from urllib.parse import urlencode

import httpx

ImageRef = dict  # {"filename": str, "subfolder": str, "type": str}


class ComfyError(Exception):
    """Base for ComfyUI backend failures."""


class ComfySubmitError(ComfyError):
    """POST /prompt was rejected before execution (graph/validation error)."""

    def __init__(self, node_errors: dict | None = None, error=None):
        self.node_errors = node_errors or {}
        self.error = error
        super().__init__(self._render())

    def _render(self) -> str:
        parts = []
        if self.error:
            parts.append(self.error.get("message") if isinstance(self.error, dict) else str(self.error))
        if self.node_errors:
            parts.append("node_errors: " + json.dumps(self.node_errors)[:800])
        return "; ".join(p for p in parts if p) or "comfy prompt validation failed"


class ComfyExecError(ComfyError):
    """A node raised during execution."""

    def __init__(self, prompt_id: str, message: str, node_type: str | None = None):
        self.prompt_id = prompt_id
        self.node_type = node_type
        super().__init__(message or "comfy execution error")


class ComfyTimeout(ComfyError):
    """The job did not finish within the wall-clock budget."""

    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        super().__init__(f"comfy job {prompt_id} timed out")


def view_query(ref: ImageRef) -> str:
    return urlencode({
        "filename": ref["filename"],
        "subfolder": ref.get("subfolder", ""),
        "type": ref.get("type", "output"),
    })


class ComfyClient:
    def __init__(self, base_url: str, http: httpx.AsyncClient, client_id: str | None = None):
        self._base = base_url.rstrip("/")
        self._http = http
        self.client_id = client_id or str(uuid.uuid4())

    async def upload_image(self, data: bytes, filename: str,
                           content_type: str = "image/png") -> tuple[str, str]:
        """Upload reference bytes to ComfyUI's input dir. Returns the server's
        (name, subfolder) — comfy may rename on collision, so never assume ours."""
        resp = await self._http.post(
            f"{self._base}/upload/image",
            files={"image": (filename, data, content_type)},
            data={"overwrite": "true", "type": "input"},
        )
        resp.raise_for_status()
        try:
            body = resp.json()
        except ValueError:
            raise ComfyError("invalid ComfyUI upload response") from None
        if not isinstance(body, dict):
            raise ComfyError("invalid ComfyUI upload response")
        name, subfolder = body.get("name"), body.get("subfolder", "")
        if not isinstance(name, str) or not name.strip() or not isinstance(subfolder, str):
            raise ComfyError("invalid ComfyUI upload response")
        return name, subfolder

    async def submit(self, graph: dict) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        resp = await self._http.post(f"{self._base}/prompt", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            if "prompt_id" in body:
                return body["prompt_id"]
            raise ComfySubmitError(body.get("node_errors"), body.get("error"))
        try:
            body = resp.json()
        except ValueError:
            raise ComfyError(f"comfy /prompt {resp.status_code}: {resp.text[:500]}")
        raise ComfySubmitError(body.get("node_errors"), body.get("error"))

    async def wait(self, prompt_id: str, timeout: float, poll_interval: float = 0.5,
                   output_kind: str = "images") -> list[ImageRef]:
        deadline = time.monotonic() + timeout
        while True:
            resp = await self._http.get(f"{self._base}/history/{prompt_id}")
            resp.raise_for_status()
            entry = resp.json().get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    node_type, msg = self._extract_error(status)
                    raise ComfyExecError(prompt_id, msg, node_type)
                if status.get("completed") or status.get("status_str") == "success" or entry.get("outputs"):
                    outputs = entry.get("outputs") or {}
                    if output_kind == "videos":
                        return self._collect_videos(outputs)
                    return self._collect_images(outputs)
            if time.monotonic() >= deadline:
                raise ComfyTimeout(prompt_id)
            await asyncio.sleep(poll_interval)

    async def history_item(self, prompt_id: str) -> dict | None:
        resp = await self._http.get(f"{self._base}/history/{prompt_id}")
        resp.raise_for_status()
        return resp.json().get(prompt_id)

    async def queue(self) -> dict:
        resp = await self._http.get(f"{self._base}/queue")
        resp.raise_for_status()
        return resp.json()

    async def fetch(self, ref: ImageRef) -> bytes:
        resp = await self._http.get(f"{self._base}/view", params={
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        })
        resp.raise_for_status()
        return resp.content

    async def object_info(self, class_name: str) -> dict:
        resp = await self._http.get(f"{self._base}/object_info/{class_name}")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _collect_images(outputs: dict) -> list[ImageRef]:
        return ComfyClient._collect_media(outputs, "images")

    @staticmethod
    def _collect_videos(outputs: dict) -> list[ImageRef]:
        refs = ComfyClient._collect_media(outputs, "videos")
        if refs:
            return refs
        refs = ComfyClient._collect_media(outputs, "gifs")
        if refs:
            return refs
        # ComfyUI's native SaveVideo node serializes PreviewVideo results as
        # animated image entries in its history response.
        refs = []
        for node_out in outputs.values():
            if node_out.get("animated"):
                refs.extend(ComfyClient._collect_media({"node": node_out}, "images"))
        return refs

    @staticmethod
    def _collect_media(outputs: dict, key: str) -> list[ImageRef]:
        refs: list[ImageRef] = []
        for node_out in outputs.values():
            for img in node_out.get(key, []):
                refs.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
        return refs

    @staticmethod
    def _extract_error(status: dict) -> tuple[str | None, str]:
        for m in status.get("messages", []):
            if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error":
                data = m[1] or {}
                msg = data.get("exception_message") or data.get("exception_type") or "execution error"
                return data.get("node_type"), msg
        return None, "execution error"
