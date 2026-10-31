"""Durable native-audio receipts, bounded FLAC caching, and backend observation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import httpx

from .comfy_client import ComfyClient, ComfyError, ComfySubmitError
from .errors import APIError
from .image_jobs import event_time, queue_index
from . import yue2_graph

JOB_TTL = 24 * 60 * 60
MAX_PENDING = 8
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_CACHE_BYTES = 512 * 1024 * 1024
TERMINAL = {"completed", "failed", "cancelled"}
logger = logging.getLogger(__name__)


async def availability(comfy) -> dict:
    graph = yue2_graph.build_text_to_music("style", "lyrics")
    classes = {node["class_type"] for node in graph.values()}
    schemas = dict(zip(sorted(classes), await asyncio.gather(
        *(comfy.object_info(name) for name in sorted(classes)))))
    missing = sorted(name for name in classes if not schemas[name].get(name))
    incompatible = set()
    for node in graph.values():
        name = node["class_type"]
        schema = schemas[name].get(name, {})
        ports = schema.get("input", {})
        declared = {**ports.get("required", {}), **ports.get("optional", {})}
        if name in missing:
            continue
        if set(node["inputs"]) - set(declared):
            incompatible.add(name)
        for field, value in node["inputs"].items():
            if not isinstance(value, list):
                continue
            source_name = graph[value[0]]["class_type"]
            outputs = schemas[source_name].get(source_name, {}).get("output", [])
            port = declared.get(field, [])
            if value[1] >= len(outputs) or not port or port[0] != outputs[value[1]]:
                incompatible.add(name)
        for field in ("mode", "format", "sampler_name", "scheduler"):
            if field not in node["inputs"]:
                continue
            port = declared.get(field, [])
            options = port[0] if port and isinstance(port[0], list) else \
                port[1].get("options", []) if len(port) > 1 and isinstance(port[1], dict) else []
            choices = [option.get("key") if isinstance(option, dict) else option for option in options]
            if node["inputs"][field] not in choices:
                incompatible.add(name)
    loader = schemas["CheckpointLoaderSimple"].get("CheckpointLoaderSimple", {})
    choices = loader.get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
    installed = isinstance(choices, list) and yue2_graph.CHECKPOINT in choices
    return {"available": not missing and not incompatible and installed, "missing_nodes": missing,
            "incompatible_nodes": sorted(incompatible), "checkpoint_available": installed}


def flac_info(content: bytes) -> dict:
    if (len(content) <= 42 or content[:4] != b"fLaC" or content[4] & 0x7f != 0
            or int.from_bytes(content[5:8], "big") != 34):
        raise ValueError("backend did not produce a FLAC stream with STREAMINFO")
    stream = int.from_bytes(content[18:26], "big")
    sample_rate = stream >> 44
    channels = ((stream >> 41) & 7) + 1
    samples = stream & ((1 << 36) - 1)
    if not sample_rate or not samples:
        raise ValueError("backend FLAC has no measured duration")
    return {"format": "flac", "mime_type": "audio/flac", "bytes": len(content),
            "duration_seconds": samples / sample_rate, "sample_rate": sample_rate,
            "channels": channels, "sha256": hashlib.sha256(content).hexdigest()}


class AudioJobStore:
    def __init__(self, directory: str):
        self.filename = Path(directory).expanduser() / "audio.sqlite3"

    @contextmanager
    def connect(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.filename, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, prompt_id TEXT, backend TEXT NOT NULL,
                metadata TEXT NOT NULL, status TEXT NOT NULL,
                created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                started_at INTEGER, completed_at INTEGER, missing_since INTEGER,
                error TEXT, output TEXT, content BLOB
            )""")
            with connection:
                yield connection
        finally:
            connection.close()

    def reserve(self, backend: str, metadata: dict) -> str:
        now = int(time.time())
        job_id = f"audio_job_{now}_{uuid.uuid4().hex}"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM jobs WHERE expires_at <= ?", (now,))
            count = connection.execute("SELECT COUNT(*) FROM jobs WHERE status NOT IN "
                                       "('completed', 'failed', 'cancelled')").fetchone()[0]
            if count >= MAX_PENDING:
                raise APIError(429, "audio queue is full; retry shortly", type="rate_limit_error",
                               code="queue_full", retry_after=5)
            connection.execute("INSERT INTO jobs (id, backend, metadata, status, created_at, "
                               "expires_at) VALUES (?, ?, ?, 'submitting', ?, ?)",
                               (job_id, backend, json.dumps(metadata), now, now + JOB_TTL))
        return job_id

    def get(self, job_id: str) -> dict:
        match = re.fullmatch(r"audio_job_([0-9]{10})_([a-f0-9]{32})", job_id)
        if not match or int(match[1]) > time.time() + 60:
            raise APIError(404, "audio job not found", code="audio_job_not_found")
        if int(match[1]) + JOB_TTL <= time.time():
            raise APIError(410, "audio job has expired", code="audio_job_expired")
        with self.connect() as connection:
            row = connection.execute("SELECT id, prompt_id, backend, metadata, status, created_at, "
                                     "expires_at, started_at, completed_at, missing_since, error, "
                                     "output FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise APIError(404, "audio job not found", code="audio_job_not_found")
        return dict(row)

    def update(self, job_id: str, **values):
        if not values or set(values) - {"prompt_id", "status", "started_at", "completed_at",
                                      "missing_since", "error"}:
            raise ValueError("invalid audio job update")
        fields = ", ".join(f"{field} = ?" for field in values)
        with self.connect() as connection:
            connection.execute(f"UPDATE jobs SET {fields} WHERE id = ? AND status NOT IN "
                               "('completed', 'failed', 'cancelled')", (*values.values(), job_id))

    def active(self) -> list[dict]:
        if not self.filename.exists():
            return []
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE expires_at <= ?", (int(time.time()),))
            ids = [row[0] for row in connection.execute("SELECT id FROM jobs WHERE status NOT IN "
                                                       "('completed', 'failed', 'cancelled')")]
        return [self.get(job_id) for job_id in ids]

    def complete(self, job_id: str, content: bytes, output: dict, started_at, completed_at):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM jobs WHERE expires_at <= ?", (int(time.time()),))
            size = connection.execute("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM jobs").fetchone()[0]
            if len(content) > MAX_OUTPUT_BYTES or size + len(content) > MAX_CACHE_BYTES:
                raise ValueError("audio output cache size limit exceeded")
            connection.execute("UPDATE jobs SET status = 'completed', content = ?, output = ?, "
                               "started_at = ?, completed_at = ?, missing_since = NULL "
                               "WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')",
                               (content, json.dumps(output), started_at, completed_at, job_id))

    def content(self, job_id: str) -> bytes:
        with self.connect() as connection:
            return connection.execute("SELECT content FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]


class AudioJobs:
    def __init__(self, directory: str, backend, backend_url: str | None):
        self.store = AudioJobStore(directory)
        self.backend = backend
        self.backend_url = (backend_url or "").rstrip("/")
        self.lock = asyncio.Lock()
        self.submissions: set[asyncio.Task] = set()

    def fail(self, job_id: str, code: str, message: str, status: str = "failed"):
        self.store.update(job_id, status=status, completed_at=int(time.time()),
                          error=json.dumps({"code": code, "message": message}))

    def describe(self, job: dict, base_url: str) -> dict:
        metadata = json.loads(job["metadata"])
        url = f"{base_url}/v1/audio/jobs/{job['id']}"
        result = {"id": job["id"], "object": "audio_job", "model": yue2_graph.MODEL_ID,
                  "status": job["status"], "created_at": job["created_at"],
                  "expires_at": job["expires_at"], "started_at": job["started_at"],
                  "completed_at": job["completed_at"], "status_url": url,
                  "seed": metadata["seed"], "planning": metadata.get("planning", "full"),
                  "recipe_version": metadata.get("recipe_version", yue2_graph.RECIPE_VERSION),
                  "max_duration_seconds": metadata["max_duration_seconds"],
                  "graph_sha256": metadata["graph_sha256"], "finish_reason": None,
                  "error": json.loads(job["error"]) if job["error"] else None}
        if job["output"]:
            result["output"] = {**json.loads(job["output"]), "audio_url": url + "/content",
                                "download_url": url + "/content?download=true"}
        return result

    async def submit(self, graph: dict, *, seed: int, max_duration: float, base_url: str,
                     queue_lock: asyncio.Lock) -> dict:
        await queue_lock.acquire()
        try:
            queue = await self.backend.queue()
            if sum(len(queue.get(key, [])) for key in ("queue_running", "queue_pending")) >= MAX_PENDING:
                raise APIError(429, "backend queue is full; retry shortly", type="rate_limit_error",
                               code="queue_full", retry_after=5)
            output_node = next(key for key, node in graph.items() if node["class_type"] == "SaveAudioAdvanced")
            metadata = {"seed": seed, "planning": "full", "recipe_version": yue2_graph.RECIPE_VERSION,
                        "max_duration_seconds": max_duration, "output_node": output_node,
                        "graph_sha256": hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()}
            job_id = self.store.reserve(self.backend_url, metadata)
            task = asyncio.create_task(self._submit_locked(job_id, graph, base_url, queue_lock))
        except BaseException:
            queue_lock.release()
            raise
        self.submissions.add(task)
        task.add_done_callback(self._submission_done)
        return await asyncio.shield(task)

    def _submission_done(self, task: asyncio.Task):
        self.submissions.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("audio submission could not persist its receipt")

    async def _submit_locked(self, job_id: str, graph: dict, base_url: str, queue_lock: asyncio.Lock):
        try:
            await self._submit(job_id, graph)
            return self.describe(self.store.get(job_id), base_url)
        finally:
            queue_lock.release()

    async def _submit(self, job_id: str, graph: dict):
        try:
            prompt_id = await self.backend.submit(graph)
        except ComfySubmitError:
            self.fail(job_id, "upstream_invalid_graph", "backend rejected the YuE2 graph")
        except (httpx.HTTPError, ComfyError, asyncio.CancelledError):
            self.store.update(job_id, status="submission_unknown", error=json.dumps({
                "code": "submission_unknown", "message": "backend acceptance is unknown; "
                "check ComfyUI before submitting again; this job will not be resubmitted"}))
        else:
            self.store.update(job_id, prompt_id=prompt_id, status="queued")

    async def refresh(self, job: dict) -> dict:
        if job["status"] in TERMINAL or self.backend is None:
            return job
        if job["backend"] != self.backend_url:
            self.fail(job["id"], "audio_backend_changed", "the accepted job's backend is no longer configured")
        elif not job["prompt_id"]:
            if job["status"] == "submitting" and time.time() - job["created_at"] >= 600:
                self.store.update(job["id"], status="submission_unknown", error=json.dumps({
                    "code": "submission_unknown", "message": "submission was interrupted; "
                    "check ComfyUI before submitting again; this job will not be resubmitted"}))
        else:
            await self._refresh_known(job)
        return self.store.get(job["id"])

    async def _refresh_known(self, job: dict):
        entry = await self.backend.history_item(job["prompt_id"])
        now = int(time.time())
        if entry:
            status = entry.get("status") or {}
            events = {message[0] for message in status.get("messages", []) if isinstance(message, list) and message}
            if "execution_interrupted" in events:
                self.fail(job["id"], "generation_cancelled", "backend execution was interrupted", "cancelled")
            elif status.get("status_str") == "error":
                code = "out_of_memory" if ComfyClient.is_out_of_memory(status) else "generation_failed"
                self.fail(job["id"], code, "backend audio generation failed")
            elif status.get("completed") or status.get("status_str") == "success":
                metadata = json.loads(job["metadata"])
                output = entry.get("outputs", {}).get(metadata["output_node"], {})
                refs = ComfyClient._collect_media({"save": output}, "audio")
                if len(refs) != 1:
                    self.fail(job["id"], "audio_output_missing", "backend did not return one saved audio output")
                    return
                try:
                    content = await self.backend.fetch(refs[0], max_bytes=MAX_OUTPUT_BYTES)
                    info = flac_info(content)
                    self.store.complete(job["id"], content, info,
                                        event_time(status, "execution_start") or job["started_at"],
                                        event_time(status, "execution_success") or now)
                except (ValueError, ComfyError) as exc:
                    self.fail(job["id"], "invalid_audio_output", str(exc))
        else:
            queue = await self.backend.queue()
            if queue_index(queue.get("queue_running", []), job["prompt_id"]) is not None:
                self.store.update(job["id"], status="in_progress", started_at=job["started_at"] or now,
                                  missing_since=None)
            elif queue_index(queue.get("queue_pending", []), job["prompt_id"]) is not None:
                self.store.update(job["id"], status="queued", missing_since=None)
            elif job["missing_since"] is None:
                self.store.update(job["id"], missing_since=now)
            elif now - job["missing_since"] >= 60:
                self.fail(job["id"], "upstream_job_lost", "job is absent from backend queue/history; it was not resubmitted")

    async def retrieve(self, job_id: str, base_url: str) -> dict:
        async with self.lock:
            job = self.store.get(job_id)
            try:
                job = await self.refresh(job)
            except (httpx.HTTPError, ComfyError):
                pass
            return self.describe(job, base_url)

    async def watch(self):
        while True:
            try:
                async with self.lock:
                    for job in self.store.active():
                        try:
                            await self.refresh(job)
                        except (httpx.HTTPError, ComfyError):
                            logger.warning("audio backend refresh deferred")
                        except APIError as exc:
                            if exc.status not in (404, 410):
                                raise
            except (sqlite3.Error, OSError):
                logger.warning("audio job refresh deferred")
            await asyncio.sleep(2)
