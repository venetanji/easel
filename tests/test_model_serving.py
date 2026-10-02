"""Retired model retrieval and concurrent per-backend queue admission."""
import asyncio
import dataclasses
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from tests.test_image_jobs import QueuedComfy


def serving_app(tmp_path, image, video=None, shared=False):
    settings = dataclasses.replace(
        Settings.from_env({}), comfy_url="http://image",
        comfy_video_url=("http://image/" if shared else "http://video") if video else None,
        image_job_dir=str(tmp_path),
    )
    return create_app(settings=settings, comfy=image, comfy_video=video)


def test_discovery_exposes_only_qwen_and_ltx_when_video_is_configured(tmp_path):
    app = serving_app(tmp_path, QueuedComfy(), QueuedComfy())
    response = TestClient(app).get("/v1/models")
    assert [model["id"] for model in response.json()["data"]] == ["qwen-image-2.1", "ltx-2.5"]


@pytest.mark.parametrize("model", ["flux2-9b", "flux2-4b", "flux-2.5"])
@pytest.mark.parametrize("endpoint", ["generations", "edits", "variations"])
def test_retired_model_requests_are_rejected_before_submission_or_upload(tmp_path, model, endpoint):
    comfy = QueuedComfy()
    client = TestClient(serving_app(tmp_path, comfy))
    fields = {"model": model, "prompt": "A red cup"}
    if endpoint == "generations":
        response = client.post("/v1/images/generations", json=fields,
                               headers={"Prefer": "respond-async"})
    else:
        response = client.post("/v1/images/" + endpoint, data=fields,
                               headers={"Prefer": "respond-async"},
                               files={"image": ("ref.png", b"REFERENCE", "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_not_found"
    assert comfy.submissions == []
    assert comfy.uploaded == []
    assert client.app.state.image_jobs.store.active() == []


@pytest.mark.parametrize("model", ["flux2-9b", "flux2-4b"])
@pytest.mark.parametrize("response_format", ["b64_json", "url"])
def test_cached_retired_flux_jobs_are_still_retrievable(tmp_path, model, response_format):
    comfy = QueuedComfy()
    app = serving_app(tmp_path, comfy)
    store = app.state.image_jobs.store
    job_id = store.reserve("image", model, response_format)
    now = int(time.time())
    store.complete(job_id, [b"OLD-FLUX-PNG"], now - 5, now)
    client = TestClient(app)
    response = client.get("/v1/images/jobs/" + job_id)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["model"] == model
    if response_format == "url":
        assert client.get(response.json()["data"][0]["url"]).content == b"OLD-FLUX-PNG"
    assert client.get("/v1/images/jobs/" + job_id + "/content/0").content == b"OLD-FLUX-PNG"
    assert comfy.submissions == []


def test_pending_retired_flux_job_is_observed_without_resubmission(tmp_path):
    comfy = QueuedComfy()
    app = serving_app(tmp_path, comfy)
    job_id = app.state.image_jobs.store.reserve("image", "flux2-9b", "b64_json")
    app.state.image_jobs.store.update(job_id, prompt_id="legacy-prompt", status="queued")
    comfy.pending = [[1, "legacy-prompt"]]
    client = TestClient(app)
    assert client.get("/v1/images/jobs/" + job_id).json()["status"] == "queued"
    comfy.finish("legacy-prompt")
    assert client.get("/v1/images/jobs/" + job_id).json()["status"] == "completed"
    assert comfy.submissions == []


class SlowQueuedComfy(QueuedComfy):
    async def queue(self):
        snapshot = await super().queue()
        snapshot = {key: list(items) for key, items in snapshot.items()}
        await asyncio.sleep(0.01)
        return snapshot

    async def submit(self, graph):
        await asyncio.sleep(0.01)
        return await super().submit(graph)


async def submit_image(client, server=None):
    fields = {"prompt": "A red cup"}
    if server is not None:
        fields["server"] = server
    return await client.post("/v1/images/generations", json=fields,
                             headers={"Prefer": "respond-async"})


async def submit_video(client):
    return await client.post("/v1/videos", data={"model": "ltx-2.5", "prompt": "A static cup",
                                               "seconds": "2", "size": "512x320"})


async def test_concurrent_mixed_admission_cannot_double_shared_backend_capacity(tmp_path):
    comfy = SlowQueuedComfy()
    app = serving_app(tmp_path, comfy, comfy, shared=True)
    assert app.state.queue_locks["image"] is app.state.queue_locks["video"]
    assert app.state.inflight is app.state.inflight_video
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        submissions = [submit_image(client, "video" if index % 2 else "image")
                       if index % 3 else submit_video(client) for index in range(12)]
        responses = await asyncio.gather(*submissions)
    assert sum(response.status_code in (200, 202) for response in responses) == 8
    rejected = [response for response in responses if response.status_code == 429]
    assert len(rejected) == 4
    assert all("Retry-After" in response.headers for response in rejected)
    assert len(comfy.submissions) == 8
    assert len(comfy.pending) == 8


async def test_distinct_backends_keep_independent_capacity_and_submission_locks(tmp_path):
    image, video = SlowQueuedComfy(), SlowQueuedComfy()
    app = serving_app(tmp_path, image, video)
    assert app.state.queue_locks["image"] is not app.state.queue_locks["video"]
    assert app.state.inflight is not app.state.inflight_video
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        responses = await asyncio.gather(*[submit_image(client) for _index in range(8)],
                                         *[submit_video(client) for _index in range(8)])
    assert all(response.status_code in (200, 202) for response in responses)
    assert len(image.submissions) == len(video.submissions) == 8


def test_shared_backend_synchronous_limit_cannot_be_bypassed_by_route_alias(tmp_path):
    comfy = QueuedComfy()
    app = serving_app(tmp_path, comfy, comfy, shared=True)
    assert app.state.inflight.acquire()
    try:
        response = TestClient(app).post("/v1/images/generations",
                                        json={"prompt": "A cup", "server": "video"})
        assert response.status_code == 429
        assert comfy.submissions == []
    finally:
        app.state.inflight.release()


async def test_admission_lock_is_released_before_synchronous_generation_finishes(tmp_path):
    started, finished = asyncio.Event(), asyncio.Event()

    class BlockingComfy(QueuedComfy):
        async def wait(self, prompt_id, timeout, poll_interval=0.5):
            started.set()
            await finished.wait()
            return [{"filename": "out.png", "type": "output"}]

    comfy = BlockingComfy()
    app = serving_app(tmp_path, comfy)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        synchronous = asyncio.create_task(client.post("/v1/images/generations",
                                                      json={"prompt": "A cup"}))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            response = await asyncio.wait_for(submit_image(client), timeout=2)
            assert response.status_code == 202
            assert len(comfy.submissions) == 2
        finally:
            finished.set()
            assert (await synchronous).status_code == 200
