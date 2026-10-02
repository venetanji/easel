"""Async image contract, durable caching, admission, and read-only polling."""
import base64
import asyncio
import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.comfy_client import ComfySubmitError
from easel.config import Settings
from easel.errors import APIError
from easel.image_jobs import IMAGE_JOB_TTL, MAX_IMAGE_JOBS, ImageJobs, ImageJobStore
from tests.test_app import FakeComfy


class QueuedComfy(FakeComfy):
    def __init__(self):
        super().__init__()
        self.submissions = []
        self.histories = {}
        self.running = []
        self.pending = []
        self.unavailable = False
        self.history_calls = []

    async def submit(self, graph):
        if self.submit_exc:
            raise self.submit_exc
        self.submitted_graph = graph
        self.submissions.append(graph)
        prompt_id = f"image-prompt-{len(self.submissions)}"
        self.pending.append([len(self.submissions), prompt_id])
        return prompt_id

    async def wait(self, *args, **kwargs):
        raise AssertionError("async acceptance must not wait for execution")

    async def queue(self):
        if self.unavailable:
            raise httpx.ConnectError("backend offline")
        return {"queue_running": self.running, "queue_pending": self.pending}

    async def history_item(self, prompt_id):
        self.history_calls.append(prompt_id)
        return self.histories.get(prompt_id)

    def finish(self, prompt_id, *, status="success", outputs=True, event="execution_error"):
        now = int(time.time())
        self.pending = [item for item in self.pending if item[1] != prompt_id]
        self.running = [item for item in self.running if item[1] != prompt_id]
        messages = [["execution_start", {"timestamp": (now - 20) * 1000}]]
        if status == "success":
            messages.append(["execution_success", {"timestamp": now * 1000}])
        else:
            messages.append([event, {"timestamp": now * 1000, "exception_message": "CUDA OOM"}])
        self.histories[prompt_id] = {
            "status": {"status_str": status, "completed": True, "messages": messages},
            "outputs": {"save": {"images": [{"filename": "out.png", "type": "output"}]}}
                if outputs else {},
        }


@pytest.fixture
def async_api(tmp_path):
    image, video = QueuedComfy(), QueuedComfy()
    settings = dataclasses.replace(Settings.from_env({}), image_job_dir=str(tmp_path),
                                   comfy_video_url="http://video", api_key="secret")
    app = create_app(settings=settings, comfy=image, comfy_video=video)
    with TestClient(app, headers={"Authorization": "Bearer secret"}) as client:
        yield client, image, video, settings


def submit(client, endpoint="generations", **options):
    fields = {"prompt": "a red wooden chair", "model": "qwen-image-2.1", **options}
    headers = {"Prefer": "respond-async"}
    if endpoint == "generations":
        return client.post("/v1/images/" + endpoint, headers=headers, json=fields)
    return client.post("/v1/images/" + endpoint, headers=headers, data=fields,
                       files={"image": ("ref.png", b"REFERENCE", "image/png")})


@pytest.mark.parametrize("endpoint", ["generations", "edits", "variations"])
def test_receipt_and_completed_data_match_client_contract(async_api, endpoint):
    client, image, _video, _settings = async_api
    response = submit(client, endpoint)
    assert response.status_code == 202
    assert response.headers["Preference-Applied"] == "respond-async"
    receipt = response.json()
    assert receipt["id"].startswith("image_job_")
    assert receipt["status"] == "queued"
    assert receipt["progress"] is None
    assert receipt["queue_position"] == 1
    assert receipt["queue_ahead"] == 0
    assert receipt["estimated_wait_seconds"] is None
    assert receipt["estimated_completion_at"] is None
    assert receipt["expires_at"] - receipt["created_at"] == IMAGE_JOB_TTL
    assert len(image.submissions) == 1
    job_id = receipt["id"]
    assert client.get(response.headers["Location"]).json()["id"] == job_id
    image.finish("image-prompt-1")
    completed = client.get("/v1/images/jobs/" + job_id).json()
    assert completed["status"] == "completed"
    assert completed["progress"] == 100
    assert completed["estimated_wait_seconds"] == 0
    assert completed["estimated_completion_at"] == completed["completed_at"]
    assert base64.b64decode(completed["data"][0]["b64_json"]) == b"PNG:out.png"
    assert len(image.submissions) == 1


def test_cached_outputs_survive_easel_and_comfy_restart(async_api):
    client, image, _video, settings = async_api
    job_id = submit(client, response_format="url").json()["id"]
    image.finish("image-prompt-1")
    completed = client.get("/v1/images/jobs/" + job_id).json()
    url = completed["data"][0]["url"]
    assert url == f"http://testserver/v1/images/jobs/{job_id}/content/0"
    empty = QueuedComfy()
    empty.unavailable = True
    app = create_app(settings=settings, comfy=empty, comfy_video=empty)
    with TestClient(app, headers={"Authorization": "Bearer secret"}) as restarted:
        assert restarted.get("/v1/images/jobs/" + job_id).json()["status"] == "completed"
        assert restarted.get(url).content == b"PNG:out.png"
        assert empty.submissions == empty.fetched == empty.history_calls == []


def test_pending_receipt_survives_easel_restart_without_resubmitting(async_api):
    client, image, video, settings = async_api
    job_id = submit(client).json()["id"]
    app = create_app(settings=settings, comfy=image, comfy_video=video)
    with TestClient(app, headers={"Authorization": "Bearer secret"}) as restarted:
        assert restarted.get("/v1/images/jobs/" + job_id).json()["status"] == "queued"
        image.finish("image-prompt-1")
        assert restarted.get("/v1/images/jobs/" + job_id).json()["status"] == "completed"
    assert len(image.submissions) == 1


def test_watcher_caches_completion_without_client_polling(async_api):
    client, image, _video, _settings = async_api
    job_id = submit(client).json()["id"]
    image.finish("image-prompt-1")
    deadline = time.monotonic() + 5
    while client.app.state.image_jobs.store.get(job_id)["status"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert client.app.state.image_jobs.store.outputs(job_id) == [b"PNG:out.png"]
    assert len(image.submissions) == 1


def test_selected_backend_is_persisted_and_polled(async_api):
    client, image, video, _settings = async_api
    job_id = submit(client, "edits", server="video").json()["id"]
    assert image.submissions == image.uploaded == []
    assert len(video.submissions) == len(video.uploaded) == 1
    video.finish("image-prompt-1")
    assert client.get("/v1/images/jobs/" + job_id).json()["status"] == "completed"
    assert image.history_calls == []


@pytest.mark.parametrize("event,expected,code", [
    ("execution_error", "failed", "upstream_execution_error"),
    ("execution_interrupted", "cancelled", "upstream_cancelled"),
])
def test_terminal_errors_are_stable(async_api, event, expected, code):
    client, image, _video, _settings = async_api
    job_id = submit(client).json()["id"]
    image.finish("image-prompt-1", status="error", outputs=False, event=event)
    for _attempt in range(3):
        body = client.get("/v1/images/jobs/" + job_id).json()
        assert body["status"] == expected
        assert body["error"]["code"] == code
        assert body["estimated_completion_at"] is None
    assert len(image.submissions) == 1


def test_success_without_images_fails(async_api):
    client, image, _video, _settings = async_api
    job_id = submit(client).json()["id"]
    image.finish("image-prompt-1", outputs=False)
    body = client.get("/v1/images/jobs/" + job_id).json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "upstream_no_output"


def test_partial_outputs_do_not_mean_completion(async_api):
    client, image, _video, _settings = async_api
    job_id = submit(client).json()["id"]
    image.histories["image-prompt-1"] = {
        "status": {"status_str": "running", "completed": False},
        "outputs": {"preview": {"images": [{"filename": "preview.png"}]}},
    }
    body = client.get("/v1/images/jobs/" + job_id).json()
    assert body["status"] == "in_progress"
    assert "data" not in body
    assert image.fetched == []


def test_auth_covers_status_and_cached_content(async_api):
    client, image, _video, settings = async_api
    job_id = submit(client).json()["id"]
    image.finish("image-prompt-1")
    client.get("/v1/images/jobs/" + job_id)
    app = create_app(settings=settings, comfy=image)
    with TestClient(app) as anonymous:
        assert anonymous.get("/v1/images/jobs/" + job_id).status_code == 401
        assert anonymous.get(f"/v1/images/jobs/{job_id}/content/0").status_code == 401
    assert client.get(f"/v1/images/jobs/{job_id}/content/../secret").status_code in (404, 400)
    assert client.get(f"/v1/images/jobs/{job_id}/content/-1").status_code == 404


def test_full_backend_queue_rejects_before_upload(async_api):
    client, image, _video, _settings = async_api
    image.pending = [[index, f"external-{index}"] for index in range(MAX_IMAGE_JOBS)]
    response = submit(client, "edits")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert image.uploaded == image.submissions == []
    assert client.app.state.image_jobs.store.active() == []


def test_store_admission_is_atomic(tmp_path):
    store = ImageJobStore(str(tmp_path))
    def reserve(_index):
        try:
            return store.reserve("image", "qwen-image-2.1", "b64_json")
        except APIError as exc:
            assert exc.status == 429
            return None
    with ThreadPoolExecutor(max_workers=12) as executor:
        ids = list(executor.map(reserve, range(20)))
    assert len([job_id for job_id in ids if job_id]) == MAX_IMAGE_JOBS
    assert len(store.active()) == MAX_IMAGE_JOBS


def test_submit_rejection_releases_reservation(async_api):
    client, image, _video, _settings = async_api
    image.submit_exc = ComfySubmitError({}, {"message": "invalid graph"})
    assert submit(client).status_code == 502
    assert client.app.state.image_jobs.store.active() == []


def test_expired_and_unknown_receipts_have_distinct_errors(async_api):
    client, _image, _video, _settings = async_api
    expired = f"image_job_{int(time.time()) - IMAGE_JOB_TTL - 1}_{'a' * 32}"
    assert client.get("/v1/images/jobs/" + expired).status_code == 410
    unknown = f"image_job_{int(time.time())}_{'a' * 32}"
    assert client.get("/v1/images/jobs/" + unknown).status_code == 404
    assert client.get("/v1/images/jobs/invalid").status_code == 404


def test_lost_backend_job_fails_without_replacement(async_api):
    client, image, _video, _settings = async_api
    job_id = submit(client).json()["id"]
    image.pending = []
    store = client.app.state.image_jobs.store
    store.update(job_id, missing_since=int(time.time()) - 61)
    body = client.get("/v1/images/jobs/" + job_id).json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "upstream_job_lost"
    assert len(image.submissions) == 1


def test_estimates_include_completion_time_and_elapsed_run(async_api):
    client, image, _video, _settings = async_api
    first = submit(client).json()["id"]
    image.finish("image-prompt-1")
    assert client.get("/v1/images/jobs/" + first).json()["average_generation_seconds"] == 20
    second = submit(client).json()["id"]
    third = submit(client).json()["id"]
    image.running, image.pending = [[2, "image-prompt-2"]], [[3, "image-prompt-3"]]
    client.app.state.image_jobs.store.update(second, started_at=int(time.time()) - 7)
    body = client.get("/v1/images/jobs/" + second).json()
    assert body["queue_position"] == 0
    assert 12 <= body["estimated_wait_seconds"] <= 13
    body = client.get("/v1/images/jobs/" + third).json()
    assert body["queue_ahead"] == 1
    assert 32 <= body["estimated_wait_seconds"] <= 33
    assert abs(body["estimated_completion_at"] - int(time.time()) - body["estimated_wait_seconds"]) <= 1


def test_backend_outage_does_not_create_new_job(async_api):
    client, image, _video, _settings = async_api
    job_id = submit(client).json()["id"]
    image.unavailable = True
    assert client.get("/v1/images/jobs/" + job_id).status_code == 502
    assert client.app.state.image_jobs.store.get(job_id)["status"] == "queued"
    assert len(image.submissions) == 1


def test_prefer_header_is_case_insensitive_and_supports_multiple_preferences(async_api):
    client, _image, _video, _settings = async_api
    response = client.post("/v1/images/generations", json={"prompt": "a chair"},
                           headers={"Prefer": "return=representation, Respond-Async; wait=0"})
    assert response.status_code == 202


async def test_expiry_during_observation_does_not_kill_watcher(tmp_path, monkeypatch):
    backend = QueuedComfy()
    service = ImageJobs(str(tmp_path), {"image": backend})
    service.store.reserve("image", "qwen-image-2.1", "url")
    async def expired(_job, _queue):
        raise APIError(410, "job expired", code="image_job_expired")
    monkeypatch.setattr(service, "refresh", expired)
    task = asyncio.create_task(service.watch())
    try:
        await asyncio.sleep(0.02)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_storage_directory_errors_have_compact_envelope(async_api, monkeypatch):
    client, image, _video, _settings = async_api
    def unavailable(*_args):
        raise OSError("storage unavailable")
    monkeypatch.setattr(client.app.state.image_jobs.store, "reserve", unavailable)
    response = submit(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "job_storage_unavailable"
    assert image.submissions == []


def test_accepted_receipt_survives_queue_metadata_outage(async_api, monkeypatch):
    client, image, _video, _settings = async_api
    original = image.submit
    async def submit_then_disconnect(graph):
        prompt_id = await original(graph)
        image.unavailable = True
        return prompt_id
    monkeypatch.setattr(image, "submit", submit_then_disconnect)
    response = submit(client)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["queue_position"] is body["queue_ahead"] is None
    assert body["running"] is body["pending"] is None
    image.unavailable = False
    assert client.get("/v1/images/jobs/" + body["id"]).json()["status"] == "queued"
    assert len(image.submissions) == 1
