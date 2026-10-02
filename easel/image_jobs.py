"""Durable async image receipts and output caching backed by ComfyUI jobs."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import httpx

from .comfy_client import ComfyClient, ComfyError
from .errors import APIError

IMAGE_JOB_TTL = 24 * 60 * 60
MAX_IMAGE_JOBS = 8
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
TERMINAL = {"completed", "failed", "cancelled"}
logger = logging.getLogger(__name__)


def wants_async(request) -> bool:
    return any(preference.strip().split(";", 1)[0].lower() == "respond-async"
               for preference in request.headers.get("prefer", "").split(","))


def event_time(status: dict, event: str) -> int | None:
    for message in status.get("messages", []):
        if isinstance(message, list) and len(message) == 2 and message[0] == event:
            timestamp = (message[1] or {}).get("timestamp")
            if isinstance(timestamp, (int, float)):
                return int(timestamp / 1000 if timestamp > 100_000_000_000 else timestamp)
    return None


def queue_index(items, prompt_id):
    return next((index for index, item in enumerate(items)
                 if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id), None)


class ImageJobStore:
    def __init__(self, directory: str):
        self.filename = Path(directory).expanduser() / "images.sqlite3"

    @contextmanager
    def connect(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.filename, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, prompt_id TEXT, server TEXT NOT NULL,
                model TEXT NOT NULL, response_format TEXT NOT NULL,
                created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                status TEXT NOT NULL, started_at INTEGER, completed_at INTEGER,
                missing_since INTEGER, error TEXT
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS outputs (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                output_index INTEGER NOT NULL, content BLOB NOT NULL,
                PRIMARY KEY(job_id, output_index)
            )""")
            with connection:
                yield connection
        finally:
            connection.close()

    def reserve(self, server: str, model: str, response_format: str) -> str:
        now = int(time.time())
        job_id = f"image_job_{now}_{uuid.uuid4().hex}"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM jobs WHERE expires_at <= ?", (now,))
            count = connection.execute("""SELECT COUNT(*) FROM jobs
                WHERE server = ? AND status NOT IN ('completed', 'failed', 'cancelled')""",
                (server,)).fetchone()[0]
            if count >= MAX_IMAGE_JOBS:
                raise APIError(429, "image queue is full; retry shortly", type="rate_limit_error",
                               code="queue_full", retry_after=5)
            connection.execute("INSERT INTO jobs (id, server, model, response_format, "
                               "created_at, expires_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (job_id, server, model, response_format, now,
                                now + IMAGE_JOB_TTL, "submitting"))
        return job_id

    def get(self, job_id: str) -> dict:
        match = re.fullmatch(r"image_job_([0-9]{10})_([a-f0-9]{32})", job_id)
        if not match or int(match[1]) > time.time() + 60:
            raise APIError(404, "image job not found", code="image_job_not_found")
        if int(match[1]) + IMAGE_JOB_TTL <= time.time():
            raise APIError(410, "image job has expired", code="image_job_expired")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise APIError(404, "image job not found", code="image_job_not_found")
        return dict(row)

    def update(self, job_id: str, **values):
        if not values or set(values) - {"prompt_id", "status", "started_at", "completed_at",
                                      "missing_since", "error"}:
            raise ValueError("invalid image job update")
        fields = ", ".join(f"{field} = ?" for field in values)
        with self.connect() as connection:
            connection.execute(f"UPDATE jobs SET {fields} WHERE id = ? "
                               "AND status NOT IN ('completed', 'failed', 'cancelled')",
                               (*values.values(), job_id))

    def discard_reservation(self, job_id: str):
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ? AND status = 'submitting'", (job_id,))

    def active(self) -> list[dict]:
        if not self.filename.exists():
            return []
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE expires_at <= ?", (int(time.time()),))
            return [dict(row) for row in connection.execute(
                "SELECT * FROM jobs WHERE status NOT IN ('completed', 'failed', 'cancelled')")]

    def complete(self, job_id: str, images: list[bytes], started_at, completed_at):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row[0] in TERMINAL:
                return
            for index, image in enumerate(images):
                connection.execute("INSERT INTO outputs VALUES (?, ?, ?)", (job_id, index, image))
            connection.execute("UPDATE jobs SET status = 'completed', started_at = ?, "
                               "completed_at = ?, missing_since = NULL WHERE id = ?",
                               (started_at, completed_at, job_id))

    def outputs(self, job_id: str) -> list[bytes]:
        with self.connect() as connection:
            return [row[0] for row in connection.execute(
                "SELECT content FROM outputs WHERE job_id = ? ORDER BY output_index", (job_id,))]

    def average_runtime(self, server: str) -> int | None:
        with self.connect() as connection:
            rows = connection.execute("SELECT completed_at - started_at FROM jobs WHERE server = ? "
                                      "AND status = 'completed' AND completed_at > started_at "
                                      "ORDER BY completed_at DESC LIMIT 20", (server,)).fetchall()
        return round(sum(row[0] for row in rows) / len(rows)) if rows else None


class ImageJobs:
    def __init__(self, directory: str, backends: dict):
        self.store = ImageJobStore(directory)
        self.backends = backends
        self.lock = asyncio.Lock()

    async def admit(self, server: str, model: str, response_format: str) -> str:
        job_id = self.store.reserve(server, model, response_format)
        try:
            queue = await self.backends[server].queue()
            if len(queue.get("queue_running", [])) + len(queue.get("queue_pending", [])) >= MAX_IMAGE_JOBS:
                raise APIError(429, "backend queue is full; retry shortly", type="rate_limit_error",
                               code="queue_full", retry_after=5)
        except BaseException:
            self.store.discard_reservation(job_id)
            raise
        return job_id

    async def submit(self, job_id: str, graph: dict) -> dict:
        job = self.store.get(job_id)
        try:
            prompt_id = await self.backends[job["server"]].submit(graph)
        except BaseException:
            self.store.discard_reservation(job_id)
            raise
        self.store.update(job_id, prompt_id=prompt_id, status="queued")
        queue = None
        try:
            queue = await self.backends[job["server"]].queue()
        except httpx.HTTPError:
            pass
        return self.describe(self.store.get(job_id), queue)

    def fail(self, job_id: str, code: str, message: str, status="failed"):
        self.store.update(job_id, status=status, completed_at=int(time.time()),
                          error=json.dumps({"code": code, "message": message}))

    async def refresh(self, job: dict, queue: dict | None) -> dict:
        if job["status"] in TERMINAL:
            return job
        comfy = self.backends.get(job["server"])
        if comfy is None:
            return job
        if job["status"] == "submitting" and time.time() - job["created_at"] < 600:
            return job
        entry = await comfy.history_item(job["prompt_id"]) if job["prompt_id"] else None
        now = int(time.time())
        if entry:
            status = entry.get("status") or {}
            started_at = event_time(status, "execution_start") or job["started_at"]
            interrupted = any(message[0] == "execution_interrupted" for message in
                              status.get("messages", []) if isinstance(message, list) and message)
            if status.get("status_str") == "error" or interrupted:
                _node, message = ComfyClient._extract_error(status)
                cancelled = interrupted or any(
                    isinstance(item, list) and len(item) == 2 and item[0] == "execution_error"
                    and (item[1] or {}).get("exception_type", "").endswith("InterruptProcessingException")
                    for item in status.get("messages", []))
                self.fail(job["id"], "upstream_cancelled" if cancelled else "upstream_execution_error",
                          "generation was interrupted" if cancelled else message,
                          "cancelled" if cancelled else "failed")
            elif status.get("completed") or status.get("status_str") == "success":
                refs = ComfyClient._collect_images(entry.get("outputs") or {})
                if not refs:
                    self.fail(job["id"], "upstream_no_output", "comfy produced no images")
                else:
                    images = []
                    byte_count = 0
                    for ref in refs:
                        image = await comfy.fetch(ref)
                        images.append(image)
                        byte_count += len(image)
                        if len(images) > 16 or byte_count > MAX_OUTPUT_BYTES:
                            self.fail(job["id"], "output_too_large", "image output exceeds cache limits")
                            break
                    else:
                        self.store.complete(job["id"], images, started_at,
                                            event_time(status, "execution_success") or now)
            else:
                self.store.update(job["id"], status="in_progress", started_at=started_at,
                                  missing_since=None)
        elif queue is not None:
            if queue_index(queue.get("queue_running", []), job["prompt_id"]) is not None:
                self.store.update(job["id"], status="in_progress",
                                  started_at=job["started_at"] or now, missing_since=None)
            elif queue_index(queue.get("queue_pending", []), job["prompt_id"]) is not None:
                self.store.update(job["id"], status="queued", missing_since=None)
            elif job["missing_since"] is None:
                self.store.update(job["id"], missing_since=now)
            elif now - job["missing_since"] >= 60:
                self.fail(job["id"], "upstream_job_lost",
                          "job is absent from backend queue/history; it was not resubmitted")
        return self.store.get(job["id"])

    def describe(self, job: dict, queue: dict | None = None, base_url: str | None = None) -> dict:
        running = (queue or {}).get("queue_running", [])
        pending = (queue or {}).get("queue_pending", [])
        running_index = queue_index(running, job["prompt_id"])
        pending_index = queue_index(pending, job["prompt_id"])
        runtime = self.store.average_runtime(job["server"])
        status = "queued" if job["status"] == "submitting" else job["status"]
        position = None
        ahead = 0 if queue is not None or status in TERMINAL else None
        estimate = 0 if status == "completed" else None
        if status not in TERMINAL:
            if running_index is not None:
                position = 0
                if runtime is not None and job["started_at"] is not None:
                    estimate = max(0, runtime - max(0, int(time.time()) - job["started_at"]))
            elif pending_index is not None:
                position = pending_index + 1
                ahead = len(running) + pending_index
                if runtime is not None:
                    remaining = runtime
                    active = self.store.active()
                    known = next((item for item in active if running and
                                  item["prompt_id"] == running[0][1] and item["started_at"]), None)
                    if known:
                        remaining = max(0, runtime - max(0, int(time.time()) - known["started_at"]))
                    estimate = runtime * (pending_index + 1) + remaining * len(running)
        result = {
            "id": job["id"], "object": "image_job", "model": job["model"], "status": status,
            "created_at": job["created_at"], "expires_at": job["expires_at"],
            "completed_at": job["completed_at"], "progress": 100 if status == "completed" else None,
            "error": json.loads(job["error"]) if job["error"] else None,
            "queue_position": position, "queue_ahead": ahead,
            "running": len(running) if queue is not None else None,
            "pending": len(pending) if queue is not None else None,
            "average_generation_seconds": runtime, "estimated_wait_seconds": estimate,
            "estimated_completion_at": job["completed_at"] if status == "completed" else
                int(time.time()) + estimate if estimate is not None else None,
        }
        if status == "completed" and base_url is not None:
            images = self.store.outputs(job["id"])
            result["created"] = job["created_at"]
            result["data"] = [
                {"url": f"{base_url}/v1/images/jobs/{job['id']}/content/{index}"}
                if job["response_format"] == "url" else {"b64_json": base64.b64encode(image).decode()}
                for index, image in enumerate(images)
            ]
        return result

    async def retrieve(self, job_id: str, base_url: str) -> dict:
        async with self.lock:
            job = self.store.get(job_id)
            queue = None
            if job["status"] not in TERMINAL and self.backends.get(job["server"]):
                queue = await self.backends[job["server"]].queue()
                job = await self.refresh(job, queue)
            return self.describe(job, queue, base_url)

    async def watch(self):
        while True:
            try:
                async with self.lock:
                    queues = {}
                    for job in self.store.active():
                        comfy = self.backends.get(job["server"])
                        if comfy is None:
                            continue
                        try:
                            backend_key = id(comfy)
                            if backend_key not in queues:
                                queues[backend_key] = await comfy.queue()
                            await self.refresh(job, queues[backend_key])
                        except (httpx.HTTPError, ComfyError) as exc:
                            logger.warning("image backend refresh deferred (%s)", type(exc).__name__)
                        except APIError as exc:
                            if exc.status not in (404, 410):
                                logger.warning("image job refresh deferred (%s)", exc.code)
            except (sqlite3.Error, OSError) as exc:
                logger.warning("image job refresh deferred (%s)", type(exc).__name__)
            await asyncio.sleep(2)
