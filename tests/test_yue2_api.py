"""Native YuE2 contract, durable audio lifecycle, and admission without inference."""
import asyncio
import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.audio_jobs import AudioJobStore, JOB_TTL, MAX_PENDING, flac_info
from easel.comfy_client import ComfyClient, ComfyError, ComfySubmitError
from easel.config import Settings
from easel.errors import APIError
from easel import yue2_graph
from tests.test_image_jobs import QueuedComfy

FIXTURES = Path(__file__).parent / "fixtures" / "yue2"
FLAC = (FIXTURES / "tone.flac").read_bytes()
BRIEF = {"model": "yue2", "style": "English, warm folk vocal", "lyrics": "[Verse]\nLight across the bay"}


class AudioComfy(QueuedComfy):
    def __init__(self):
        super().__init__()
        self.schemas = json.loads((FIXTURES / "native_schema.json").read_text())
        self.schemas["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0] = [yue2_graph.CHECKPOINT]
        self.audio = FLAC
        self.fetched = []

    async def object_info(self, name):
        if self.unavailable:
            raise httpx.ConnectError("offline")
        return {name: self.schemas[name]} if name in self.schemas else {}

    async def fetch(self, ref, *, max_bytes=None):
        self.fetched.append(ref)
        if max_bytes is not None and len(self.audio) > max_bytes:
            raise ComfyError("output exceeds cache size limit")
        return self.audio

    def finish_audio(self, prompt_id="image-prompt-1", **kwargs):
        self.finish(prompt_id, **kwargs)
        output_node = next(key for key, node in self.submissions[-1].items()
                           if node["class_type"] == "SaveAudioAdvanced")
        self.histories[prompt_id]["outputs"] = {output_node: {"audio": [
            {"filename": "native.flac", "subfolder": "private", "type": "output"}]}}


def build(tmp_path, backend=None, **settings):
    backend = backend or AudioComfy()
    config = dataclasses.replace(Settings.from_env({}), comfy_audio_url="http://audio",
                                 image_job_dir=str(tmp_path), **settings)
    app = create_app(settings=config, comfy=backend, comfy_audio=backend)
    return TestClient(app), backend


def test_vendored_full_abc_recipe_matches_creative_graph_snapshot():
    graph = yue2_graph.build_text_to_music(BRIEF["style"], BRIEF["lyrics"])
    reference = json.loads((FIXTURES / "full_abc_graph.json").read_text())
    renumbered = {str(int(node_id) + 1): {
        **node, "inputs": {name: [str(int(value[0]) + 1), value[1]]
                           if isinstance(value, list) else value
                           for name, value in node["inputs"].items()}}
        for node_id, node in reference.items()}
    assert graph == renumbered
    music_id = next(key for key, node in graph.items() if node["class_type"] == "YuE2GenerateMusic")
    latent = next(node for node in graph.values() if node["class_type"] == "EmptyYuE2LatentAudio")
    assert latent["inputs"]["seconds"] == [music_id, 1]


def test_discovery_is_independent_of_suno_and_checks_native_readiness(tmp_path):
    client, comfy = build(tmp_path)
    assert [item["id"] for item in client.get("/v1/models").json()["data"]] == [
        "qwen-image-2.1", "moss-image-base", "moss-image-sft-8step", "yue2"]
    response = client.get("/v1/audio/capabilities").json()
    assert response["available"] and response["configured"]
    assert response["planning"] == ["full"] and response["formats"] == ["flac"]
    assert response["max_duration_seconds"]["maximum"] == 300
    assert response["seed"]["maximum"] == 2**53 - 1
    assert comfy.submissions == []
    assert client.get("/v1/audio/models").status_code == 503


def test_disabled_audio_has_no_yue2_discovery_or_submission(tmp_path):
    config = dataclasses.replace(Settings.from_env({}), image_job_dir=str(tmp_path))
    client = TestClient(create_app(settings=config, comfy=AudioComfy()))
    assert "yue2" not in [item["id"] for item in client.get("/v1/models").json()["data"]]
    assert client.post("/v1/audio/generations", json=BRIEF).status_code == 503
    assert client.get("/v1/audio/capabilities").json()["configured"] is False


@pytest.mark.parametrize("changes", [
    {"style": " "}, {"lyrics": "\x00bad"}, {"lyrics": ""}, {"style": "x" * 1001},
    {"lyrics": "x" * 12001}, {"style": 42}, {"max_duration_seconds": 0},
    {"max_duration_seconds": 300.04}, {"max_duration_seconds": "30"},
    {"max_duration_seconds": True}, {"seed": -1}, {"seed": 2**53}, {"seed": 1.5},
    {"seed": True}, {"seed": "0"}, {"dry_run": True}, {"prompt": "style"},
    {"planning": "off"}, {"abc": "X:1"}, {"filename_prefix": "../bad"}, {"model": "yue2-3b"},
])
def test_invalid_native_requests_never_submit(tmp_path, changes):
    client, comfy = build(tmp_path)
    response = client.post("/v1/audio/generations", json={**BRIEF, **changes})
    assert response.status_code == 400, response.text
    assert comfy.submissions == []
    assert client.app.state.audio_jobs.store.active() == []


@pytest.mark.parametrize("field", ["max_duration_seconds", "seed"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_numeric_requests_fail_without_submission(tmp_path, field, value):
    client, comfy = build(tmp_path)
    text = json.dumps(BRIEF)[:-1] + f', "{field}": {value}' + "}"
    assert client.post("/v1/audio/generations", content=text,
                       headers={"Content-Type": "application/json"}).status_code == 400
    assert comfy.submissions == []


@pytest.mark.parametrize("missing", ["YuE2GenerateMusic", "SaveAudioAdvanced", "checkpoint"])
def test_missing_assets_fail_before_admission(tmp_path, missing):
    client, comfy = build(tmp_path)
    if missing == "checkpoint":
        comfy.schemas["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0] = []
    else:
        comfy.schemas.pop(missing)
    assert client.get("/v1/audio/capabilities").json()["available"] is False
    response = client.post("/v1/audio/generations", json=BRIEF)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "audio_model_unavailable"
    assert comfy.submissions == [] and client.app.state.audio_jobs.store.active() == []


@pytest.mark.parametrize("mutation", ["port", "output", "mode", "format"])
def test_incompatible_native_contract_is_unavailable_before_submit(tmp_path, mutation):
    client, comfy = build(tmp_path)
    if mutation == "port":
        comfy.schemas["EmptyYuE2LatentAudio"]["input"]["required"].pop("seconds")
    elif mutation == "output":
        comfy.schemas["YuE2GenerateMusic"]["output"][1] = "INT"
    elif mutation == "mode":
        comfy.schemas["YuE2GenerateABC"]["input"]["required"]["mode"][1]["options"] = ["melody"]
    else:
        comfy.schemas["SaveAudioAdvanced"]["input"]["required"]["format"][1]["options"] = []
    assert client.get("/v1/audio/capabilities").json()["available"] is False
    assert client.post("/v1/audio/generations", json=BRIEF).status_code == 503
    assert comfy.submissions == []


def test_acceptance_completion_cached_output_and_restart(tmp_path):
    client, comfy = build(tmp_path)
    response = client.post("/v1/audio/generations", json={**BRIEF, "seed": 0, "max_duration_seconds": 300})
    assert response.status_code == 202
    receipt = response.json()
    assert receipt["status"] == "queued" and receipt["model"] == "yue2"
    assert response.headers["location"] == receipt["status_url"]
    assert receipt["seed"] == 0 and receipt["max_duration_seconds"] == 300
    path = "/v1/audio/jobs/" + receipt["id"]
    assert client.get(path + "/content").status_code == 409
    comfy.finish_audio()
    result = client.get(path).json()
    assert result["status"] == "completed" and result["finish_reason"] is None
    assert result["output"]["duration_seconds"] == 0.04
    assert result["output"]["sample_rate"] == 48000 and result["output"]["channels"] == 2
    assert "native.flac" not in json.dumps(result) and "private" not in json.dumps(result)
    assert client.get(result["output"]["audio_url"]).content == FLAC
    restarted, _other = build(tmp_path)
    comfy.unavailable = True
    assert restarted.get(path).json() == result
    content = restarted.get(path + "/content?download=true")
    assert content.content == FLAC and content.headers["content-type"] == "audio/flac"
    assert "attachment" in content.headers["content-disposition"]
    assert content.headers["content-disposition"].endswith('.flac"')
    assert len(comfy.submissions) == 1


def test_pending_job_survives_restart_without_resubmission(tmp_path):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    restarted, _backend = build(tmp_path, backend=comfy)
    assert restarted.get(receipt["status_url"]).json()["status"] == "queued"
    comfy.finish_audio()
    assert restarted.get(receipt["status_url"]).json()["status"] == "completed"
    assert len(comfy.submissions) == 1


@pytest.mark.parametrize("range_header,expected", [
    ("bytes=0-3", FLAC[:4]), ("bytes=4-", FLAC[4:]), ("bytes=-7", FLAC[-7:]),
    ("bytes=0-999999", FLAC),
])
def test_flac_ranges_and_conditionals(tmp_path, range_header, expected):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    comfy.finish_audio()
    url = receipt["status_url"] + "/content"
    response = client.get(url, headers={"Range": range_header})
    assert response.status_code == 206 and response.content == expected
    etag = response.headers["etag"]
    assert client.get(url, headers={"If-None-Match": etag}).status_code == 304
    assert client.get(url, headers={"If-None-Match": "W/" + etag}).status_code == 304
    assert client.get(url, headers={"If-None-Match": "*"}).status_code == 304
    assert client.get(url, headers={"Range": "bytes=0-3", "If-Range": '"changed"'}).content == FLAC


@pytest.mark.parametrize("selected", ["bytes=999999-", "bytes=4-2", "bytes=-0", "bytes=-", "bytes=0-1,3-4",
                                      "bytes=" + "9" * 5000 + "-"])
def test_invalid_ranges_return_416(tmp_path, selected):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    comfy.finish_audio()
    response = client.get(receipt["status_url"] + "/content", headers={"Range": selected})
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(FLAC)}"


def test_all_native_audio_routes_use_existing_auth_policy(tmp_path):
    client, comfy = build(tmp_path, api_key="secret", trusted_client_networks=())
    for path in ("/v1/audio/capabilities", "/v1/audio/jobs/invalid", "/v1/audio/jobs/invalid/content"):
        assert client.get(path).status_code == 401
    assert client.post("/v1/audio/generations", json=BRIEF).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    receipt = client.post("/v1/audio/generations", json=BRIEF, headers=headers).json()
    comfy.finish_audio()
    assert client.get(receipt["status_url"] + "/content", headers=headers).content == FLAC


@pytest.mark.parametrize("failure,expected", [
    (httpx.ReadTimeout("acceptance reply lost"), "submission_unknown"),
    (ComfySubmitError({}, "invalid graph"), "failed"),
])
def test_submission_errors_are_durable_and_never_retried(tmp_path, failure, expected):
    client, comfy = build(tmp_path)
    comfy.submit_exc = failure
    response = client.post("/v1/audio/generations", json=BRIEF)
    assert response.status_code == 202
    assert response.json()["status"] == expected
    restarted, _backend = build(tmp_path, backend=comfy)
    result = restarted.get(response.json()["status_url"]).json()
    assert result["status"] == expected and result["error"]
    assert comfy.submissions == []


def test_backend_outage_does_not_turn_queued_job_into_failure(tmp_path):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    comfy.unavailable = True
    assert client.get(receipt["status_url"]).json()["status"] == "queued"
    assert client.post("/v1/audio/generations", json=BRIEF).status_code == 503


@pytest.mark.parametrize("kind", ["missing", "other-node", "invalid", "oversized", "error", "cancelled"])
def test_terminal_failures_do_not_report_completed_audio(tmp_path, kind, monkeypatch):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    if kind in ("error", "cancelled"):
        comfy.finish_audio(status="error", event="execution_interrupted" if kind == "cancelled" else "execution_error")
    else:
        comfy.finish_audio()
    if kind == "missing":
        comfy.histories["image-prompt-1"]["outputs"] = {}
    elif kind == "other-node":
        comfy.histories["image-prompt-1"]["outputs"] = {"other": {"audio": [{"filename": "bad.flac"}]}}
    elif kind == "invalid":
        comfy.audio = b"not a FLAC"
    elif kind == "oversized":
        monkeypatch.setattr("easel.audio_jobs.MAX_OUTPUT_BYTES", 8)
    result = client.get(receipt["status_url"]).json()
    assert result["status"] == ("cancelled" if kind == "cancelled" else "failed")
    assert result["error"] and "output" not in result


def test_lost_history_and_interrupted_reservations_are_not_resubmitted(tmp_path):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    comfy.pending = []
    store = client.app.state.audio_jobs.store
    store.update(receipt["id"], missing_since=int(time.time()) - 61)
    assert client.get(receipt["status_url"]).json()["error"]["code"] == "upstream_job_lost"
    reserved = store.reserve("http://audio", {"seed": 0, "max_duration_seconds": 30, "graph_sha256": "test"})
    with store.connect() as connection:
        connection.execute("UPDATE jobs SET created_at = created_at - 601 WHERE id = ?", (reserved,))
    assert client.get("/v1/audio/jobs/" + reserved).json()["status"] == "submission_unknown"
    assert len(comfy.submissions) == 1


def test_invalid_unknown_and_expired_job_ids(tmp_path, monkeypatch):
    client, _comfy = build(tmp_path)
    assert client.get("/v1/audio/jobs/invalid").status_code == 404
    unknown = f"audio_job_{int(time.time())}_{'0' * 32}"
    assert client.get("/v1/audio/jobs/" + unknown).status_code == 404
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    future = time.time() + JOB_TTL + 1
    monkeypatch.setattr("easel.audio_jobs.time.time", lambda: future)
    assert client.get(receipt["status_url"]).status_code == 410
    assert client.get(receipt["status_url"] + "/content").status_code == 410


def test_audio_cache_is_bounded(tmp_path, monkeypatch):
    client, comfy = build(tmp_path)
    receipt = client.post("/v1/audio/generations", json=BRIEF).json()
    comfy.finish_audio()
    monkeypatch.setattr("easel.audio_jobs.MAX_CACHE_BYTES", len(FLAC) - 1)
    assert client.get(receipt["status_url"]).json()["status"] == "failed"


def test_store_reservations_are_atomic(tmp_path):
    store = AudioJobStore(str(tmp_path))

    def reserve(index):
        try:
            return store.reserve("backend", {"seed": index})
        except APIError as exc:
            assert exc.status == 429
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        ids = list(executor.map(reserve, range(20)))
    assert len([job_id for job_id in ids if job_id]) == MAX_PENDING


async def test_shared_backend_admission_remains_bounded(tmp_path):
    class SlowAudio(AudioComfy):
        async def submit(self, graph):
            await asyncio.sleep(0.01)
            return await super().submit(graph)

    comfy = SlowAudio()
    config = dataclasses.replace(Settings.from_env({}), comfy_url="http://shared",
                                 comfy_video_url="http://shared/", comfy_audio_url="http://shared/",
                                 image_job_dir=str(tmp_path))
    app = create_app(settings=config, comfy=comfy, comfy_video=comfy, comfy_audio=comfy)
    assert app.state.queue_locks["image"] is app.state.queue_locks["audio"] is app.state.queue_locks["video"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        requests = [client.post("/v1/audio/generations", json=BRIEF) if index % 2 else
                    client.post("/v1/images/generations", json={"prompt": "cup"},
                                headers={"Prefer": "respond-async"}) for index in range(12)]
        responses = await asyncio.gather(*requests)
    assert sum(response.status_code == 202 for response in responses) == MAX_PENDING
    assert all(response.status_code in (202, 429) for response in responses)
    assert len(comfy.submissions) == MAX_PENDING


async def test_disconnect_preserves_backend_acceptance_and_receipt(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowAudio(AudioComfy):
        async def submit(self, graph):
            entered.set()
            await release.wait()
            return await super().submit(graph)

    client, comfy = build(tmp_path, backend=SlowAudio())
    app = client.app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http:
        pending = asyncio.create_task(http.post("/v1/audio/generations", json=BRIEF))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert app.state.queue_locks["audio"].locked()
        release.set()
        await asyncio.gather(*app.state.audio_jobs.submissions)
    jobs = app.state.audio_jobs.store.active()
    assert len(jobs) == 1 and jobs[0]["status"] == "queued"
    assert jobs[0]["prompt_id"] == "image-prompt-1" and len(comfy.submissions) == 1


async def test_disconnect_before_admission_does_not_submit_late_work(tmp_path):
    client, comfy = build(tmp_path)
    app = client.app
    lock = app.state.queue_locks["audio"]
    await lock.acquire()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http:
            pending = asyncio.create_task(http.post("/v1/audio/generations", json=BRIEF))
            await asyncio.sleep(0.01)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
    finally:
        lock.release()
    assert app.state.audio_jobs.store.active() == [] and comfy.submissions == []
    assert app.state.audio_jobs.submissions == set()


@pytest.mark.parametrize("after_reservation", [False, True])
async def test_real_http_disconnect_respects_admission_boundary(tmp_path, after_reservation):
    entered, release, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowAudio(AudioComfy):
        async def submit(self, graph):
            entered.set()
            await release.wait()
            return await super().submit(graph)

    client, comfy = build(tmp_path, backend=SlowAudio())
    app = client.app
    lock = app.state.queue_locks["audio"]
    if not after_reservation:
        await lock.acquire()
    received_body = False
    sent = []

    async def receive():
        nonlocal received_body
        if not received_body:
            received_body = True
            return {"type": "http.request", "body": json.dumps(BRIEF).encode(), "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/v1/audio/generations",
             "raw_path": b"/v1/audio/generations", "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 1234), "server": ("test", 80)}
    task = asyncio.create_task(app(scope, receive, send))
    if after_reservation:
        await entered.wait()
    else:
        await asyncio.sleep(0.01)
    disconnected.set()
    await asyncio.wait_for(task, 1)
    assert next(message for message in sent if message["type"] == "http.response.start")["status"] == 499
    if after_reservation:
        assert lock.locked()
        release.set()
        await asyncio.gather(*app.state.audio_jobs.submissions)
        jobs = app.state.audio_jobs.store.active()
        assert len(jobs) == 1 and jobs[0]["status"] == "queued"
        assert len(comfy.submissions) == 1
    else:
        lock.release()
        assert app.state.audio_jobs.store.active() == [] and comfy.submissions == []


async def test_lifespan_reuses_shared_client_and_background_caches_audio(tmp_path):
    comfy = AudioComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_url="http://shared",
                                    comfy_audio_url="http://shared/", image_job_dir=str(tmp_path))
    app = create_app(settings=settings, comfy=comfy)
    async with app.router.lifespan_context(app):
        assert app.state.comfy_audio is app.state.comfy
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http:
            receipt = (await http.post("/v1/audio/generations", json=BRIEF)).json()
            comfy.finish_audio()
            for _attempt in range(40):
                if app.state.audio_jobs.store.get(receipt["id"])["status"] == "completed":
                    break
                await asyncio.sleep(0.1)
            assert app.state.audio_jobs.store.get(receipt["id"])["status"] == "completed"
            comfy.histories.clear()
            assert (await http.get(receipt["status_url"] + "/content")).content == FLAC


def test_flac_metadata_uses_actual_sample_count():
    assert flac_info(FLAC)["duration_seconds"] == 0.04
    assert flac_info(FLAC)["channels"] == 2


async def test_client_bounded_fetch_rejects_oversized_audio():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=FLAC))) as http:
        comfy = ComfyClient("http://comfy", http)
        with pytest.raises(ComfyError, match="size limit"):
            await comfy.fetch({"filename": "out.flac"}, max_bytes=8)
        assert await comfy.fetch({"filename": "out.flac"}, max_bytes=len(FLAC)) == FLAC


async def test_comfy_wait_can_collect_native_audio_history():
    history = {"prompt": {"status": {"status_str": "success"},
                           "outputs": {"8": {"audio": [{"filename": "out.flac"}]}}}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=history))) as http:
        outputs = await ComfyClient("http://comfy", http).wait("prompt", timeout=1, output_kind="audio")
    assert outputs == [{"filename": "out.flac", "subfolder": "", "type": "output"}]
