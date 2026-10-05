"""Audio routes and REST transport against a fake Suno server; no credits spent."""
import dataclasses
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel.suno_client import SunoClient
from tests.test_image_jobs import QueuedComfy

SONG_ID = "a8627b92-354b-40d9-a50e-ab241e4fb3a1"
ATTEMPT_ID = "69d70c38-6802-4281-a0ae-fb6f8473f715"


class AudioBytes(httpx.AsyncByteStream):
    def __init__(self, data=b"SAVED-AUDIO"):
        self.data = data
        self.closed = False

    async def __aiter__(self):
        yield self.data

    async def aclose(self):
        self.closed = True


class FakeREST:
    def __init__(self):
        self.calls = []
        self.responses = {}
        self.stream = AudioBytes()

    async def handle(self, request):
        self.calls.append(request)
        path = request.url.path.removeprefix("/api/v1/")
        if path in self.responses:
            result = self.responses[path]
            if isinstance(result, Exception):
                raise result
            if isinstance(result, httpx.Response):
                return result
            status, data = result
            return httpx.Response(status, json=data)
        if path.endswith("/audio"):
            return httpx.Response(200, stream=self.stream, headers={
                "Content-Type": "audio/mp4", "Content-Length": str(len(self.stream.data)),
                "Content-Disposition": f'inline; filename="{SONG_ID}.m4a"',
                "Accept-Ranges": "bytes",
            })
        if path.endswith("/download"):
            return httpx.Response(200, json={
                "song_id": SONG_ID, "path": f"/downloads/{SONG_ID}.m4a",
                "filename": f"{SONG_ID}.m4a", "format": "m4a", "size": 11,
                "stream_url": "http://private-backend/audio/file.m4a",
                "audio_url": "http://private-backend/api/v1/songs/file/audio",
            })
        if path.startswith("songs/"):
            return httpx.Response(200, json={"id": SONG_ID, "status": "complete", "audio_url": "http://cdn/file"})
        if path == "generation/abandon":
            return httpx.Response(200, json={"status": "abandoned", "attempt_id": ATTEMPT_ID})
        if path == "generation/status":
            return httpx.Response(200, json={"status": "idle", "metrics": {"attempts": 0}})
        return httpx.Response(202, json={"status": "submitted", "attempt_id": ATTEMPT_ID,
                                         "songs": [{"id": SONG_ID, "status": "queued"}]})


@pytest.fixture
async def audio_client(tmp_path):
    backend = FakeREST()
    settings = dataclasses.replace(Settings.from_env({}), suno_url="http://suno",
                                   suno_api_token="upstream-secret", image_job_dir=str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend.handle)) as http:
        suno = SunoClient(settings.suno_url, http, token=settings.suno_api_token)
        app = create_app(settings=settings, comfy=QueuedComfy(), suno=suno)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://easel") as client:
            yield client, backend


async def test_audio_discovery_is_optional_and_independent_of_video(audio_client):
    client, _backend = audio_client
    response = await client.get("/v1/models")
    assert [model["id"] for model in response.json()["data"]] == [
        "qwen-image-2.1", "suno-music", "suno-speech", "suno-sound",
    ]
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, api_key="client-secret")
    app.state.settings = dataclasses.replace(app.state.settings, suno_url=None)
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano"},
                                 headers={"Authorization": "Bearer client-secret"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "audio_server_unavailable"


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/generations", {"prompt": "Piano"}),
    ("GET", "/generations/status", None),
    ("POST", "/generations/abandon", {"attempt_id": ATTEMPT_ID}),
    ("GET", f"/tracks/{SONG_ID}", None),
    ("POST", f"/tracks/{SONG_ID}/download", None),
    ("GET", f"/tracks/{SONG_ID}/content", None),
])
async def test_every_audio_route_requires_easel_auth(audio_client, method, path, body):
    client, backend = audio_client
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, api_key="client-secret")
    response = await client.request(method, "/v1/audio" + path, json=body)
    assert response.status_code == 401
    assert backend.calls == []
    response = await client.request(method, "/v1/audio" + path, json=body,
                                    headers={"Authorization": "Bearer client-secret"})
    assert response.status_code in (200, 202)
    assert backend.calls[-1].headers["Authorization"] == "Bearer upstream-secret"
    assert "upstream-secret" not in response.text


@pytest.mark.parametrize("body,kind,expected", [
    ({"prompt": "Warm piano", "suno_model": "native-picker-label", "make_instrumental": True,
      "duration_seconds": 20}, "music", {"tags": "Warm piano", "model": "native-picker-label",
                                         "make_instrumental": True, "duration_seconds": 20}),
    ({"lyrics": "A song about rain", "reference_audio_id": SONG_ID, "audio_mode": "extend",
      "audio_influence": 30}, "music", {"lyrics": "A song about rain", "reference_audio_id": SONG_ID,
                                         "audio_mode": "extend", "audio_influence": 30}),
    ({"model": "suno-speech", "prompt": "Hello world", "tone": "Warm", "variety": 2},
     "speech", {"script": "Hello world", "tone": "Warm", "variety": 2}),
    ({"model": "suno-sound", "prompt": "Quiet ocean waves", "sound_type": "loop", "bpm": 80},
     "sound", {"prompt": "Quiet ocean waves", "sound_type": "loop", "bpm": 80}),
])
async def test_generation_routes_and_maps_native_options(audio_client, body, kind, expected):
    client, backend = audio_client
    response = await client.post("/v1/audio/generations", json=body)
    assert response.status_code == 202
    result = response.json()
    assert result["object"] == "audio.generation"
    assert result["model"] == body.get("model", "suno-music")
    assert result["attempt_id"] == ATTEMPT_ID
    assert result["songs"][0]["audio_url"] == f"http://easel/v1/audio/tracks/{SONG_ID}/content"
    assert response.headers["Location"] == result["status_url"]
    assert response.headers["Retry-After"] == "5"
    request = backend.calls[0]
    assert request.url.path == f"/api/v1/generate/{kind}"
    payload = json.loads(request.content)
    assert {key: payload[key] for key in expected} == expected
    assert payload.get("model") not in ("suno-music", "suno-speech", "suno-sound")
    assert "suno_model" not in payload
    if kind != "sound":
        assert "prompt" not in payload


@pytest.mark.parametrize("body", [
    {}, {"prompt": "   "}, {"model": "unknown", "prompt": "Piano"},
    {"prompt": "Piano", "weirdness": 101}, {"prompt": "Piano", "variety": -1},
    {"prompt": "Piano", "duration_seconds": 9}, {"prompt": "Piano", "duration_seconds": 361},
    {"prompt": "Piano", "reference_audio_id": "not-a-uuid"},
    {"prompt": "Piano", "reference_audio_id": SONG_ID, "inspiration_ids": [ATTEMPT_ID]},
    {"lyrics": "Lyrics", "make_instrumental": True},
    {"prompt": "Piano", "audio_influence": 30},
    {"prompt": "Piano", "inspiration_ids": [SONG_ID], "inspiration_playlist": "Favorites"},
    {"prompt": "Piano", "inspiration_ids": [SONG_ID, SONG_ID]},
    {"prompt": "Piano", "unexpected": True},
    {"model": "suno-speech", "prompt": ""}, {"model": "suno-speech", "prompt": "   "},
    {"model": "suno-speech", "prompt": "Hi", "lyrics": "not supported"},
    {"model": "suno-sound", "prompt": "Boom", "bpm": 301},
    {"model": "suno-sound", "prompt": "Boom", "sound_type": "music"},
    {"model": "suno-sound", "prompt": "Boom", "make_instrumental": True},
])
async def test_invalid_audio_requests_never_reach_suno(audio_client, body):
    client, backend = audio_client
    response = await client.post("/v1/audio/generations", json=body)
    assert response.status_code == 400
    assert backend.calls == []


async def test_dry_run_prepares_without_an_extra_submission(audio_client):
    client, backend = audio_client
    backend.responses["generate/music"] = (200, {"status": "prepared", "fields": {}})
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano", "dry_run": True})
    assert response.status_code == 200
    assert response.json()["status"] == "prepared"
    assert "Location" not in response.headers
    assert len(backend.calls) == 1
    assert json.loads(backend.calls[0].content)["dry_run"] is True


async def test_captcha_and_status_preserve_recovery_information_without_resubmitting(audio_client):
    client, backend = audio_client
    challenge = {"status": "captcha_required", "attempt_id": ATTEMPT_ID,
                 "novnc_url": "http://suno/vnc.html", "message": "Solve manually; do not retry",
                 "metrics": {"captcha_challenges": 1}}
    backend.responses["generate/music"] = (202, challenge)
    backend.responses["generation/status"] = (200, challenge)
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano"})
    result = (await client.get(response.json()["status_url"])).json()
    assert all(result[key] == value for key, value in challenge.items())
    assert [request.method for request in backend.calls] == ["POST", "GET"]


async def test_pending_conflict_keeps_attempt_for_recovery(audio_client):
    client, backend = audio_client
    generation = {"status": "pending", "attempt_id": ATTEMPT_ID}
    backend.responses["generate/music"] = (409, {"error": {
        "code": "generation_pending", "message": "awaiting confirmation", "generation": generation,
    }})
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano"})
    assert response.status_code == 409
    assert response.json()["error"]["generation"] == generation
    response = await client.post("/v1/audio/generations/abandon", json={"attempt_id": ATTEMPT_ID})
    assert response.json()["status"] == "abandoned"
    assert json.loads(backend.calls[-1].content) == {"attempt_id": ATTEMPT_ID}


@pytest.mark.parametrize("upstream,expected", [(401, 502), (403, 502), (422, 400), (503, 503), (504, 504), (500, 502)])
async def test_backend_errors_are_mapped_without_retry(audio_client, upstream, expected):
    client, backend = audio_client
    backend.responses["generate/music"] = (upstream, {"error": {"code": "test_error", "message": "failed"}})
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano"})
    assert response.status_code == expected
    assert len(backend.calls) == 1


@pytest.mark.parametrize("error,expected", [(httpx.ReadTimeout("ambiguous"), 504), (httpx.ConnectError("offline"), 502)])
async def test_ambiguous_transport_failures_never_retry_generation(audio_client, error, expected):
    client, backend = audio_client
    backend.responses["generate/music"] = error
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano"})
    assert response.status_code == expected
    assert "check generation status before retrying" in response.json()["error"]["message"]
    assert len(backend.calls) == 1


@pytest.mark.parametrize("upstream", [
    httpx.Response(200, text="not JSON"), httpx.Response(200, json=[]),
    httpx.Response(307, headers={"Location": "http://other/generate"}, json={}),
])
async def test_invalid_or_redirect_responses_are_not_retried(audio_client, upstream):
    client, backend = audio_client
    backend.responses["generate/music"] = upstream
    response = await client.post("/v1/audio/generations", json={"prompt": "Piano"})
    assert response.status_code == 502
    assert len(backend.calls) == 1


async def test_track_and_download_urls_stay_on_easel_and_hide_backend_paths(audio_client):
    client, backend = audio_client
    track = (await client.get(f"/v1/audio/tracks/{SONG_ID}")).json()
    result = (await client.post(track["download_url"])).json()
    assert result["format"] == "m4a"
    assert result["filename"] == f"{SONG_ID}.m4a"
    assert result["audio_url"] == track["audio_url"]
    assert "path" not in result and "stream_url" not in result
    assert "private-backend" not in json.dumps(result)
    assert len(backend.calls) == 2


async def test_saved_audio_streams_without_changing_its_format_and_closes_upstream(audio_client):
    client, backend = audio_client
    response = await client.get(f"/v1/audio/tracks/{SONG_ID}/content?download=true")
    assert response.status_code == 200
    assert response.content == b"SAVED-AUDIO"
    assert response.headers["Content-Type"] == "audio/mp4"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert ".m4a" in response.headers["Content-Disposition"]
    assert backend.calls[-1].url.params["download"] == "true"
    assert backend.stream.closed


async def test_byte_range_requests_and_response_headers_are_forwarded(audio_client):
    client, backend = audio_client
    stream = AudioBytes(b"SAVE")
    backend.responses[f"songs/{SONG_ID}/audio"] = httpx.Response(206, stream=stream, headers={
        "Content-Type": "audio/mpeg", "Content-Range": "bytes 0-3/11", "Content-Length": "4",
    })
    response = await client.get(f"/v1/audio/tracks/{SONG_ID}/content", headers={"Range": "bytes=0-3"})
    assert response.status_code == 206
    assert response.content == b"SAVE"
    assert response.headers["Content-Range"] == "bytes 0-3/11"
    assert backend.calls[-1].headers["Range"] == "bytes=0-3"
    assert stream.closed


@pytest.mark.parametrize("status,data,expected", [
    (404, {"error": {"code": "audio_not_downloaded", "message": "Save audio first"}}, 404),
    (200, {"not": "audio"}, 502),
])
async def test_content_rejects_unsaved_or_invalid_audio(audio_client, status, data, expected):
    client, backend = audio_client
    backend.responses[f"songs/{SONG_ID}/audio"] = (status, data)
    response = await client.get(f"/v1/audio/tracks/{SONG_ID}/content")
    assert response.status_code == expected


async def test_invalid_track_ids_and_abandon_ids_do_not_reach_backend(audio_client):
    client, backend = audio_client
    for suffix in ("", "/content", "/download"):
        method = "POST" if suffix == "/download" else "GET"
        response = await client.request(method, "/v1/audio/tracks/not-a-uuid" + suffix)
        assert response.status_code == 400
    response = await client.post("/v1/audio/generations/abandon", json={"attempt_id": "invalid"})
    assert response.status_code == 400
    assert backend.calls == []


def test_configured_suno_client_is_started_and_closed_by_app_lifespan(tmp_path):
    settings = dataclasses.replace(Settings.from_env({}), suno_url="http://suno",
                                   suno_timeout=92, image_job_dir=str(tmp_path))
    app = create_app(settings=settings, comfy=QueuedComfy())
    with TestClient(app):
        assert isinstance(app.state.suno, SunoClient)
        assert app.state.suno.timeout == 92
        assert not app.state.suno.http.is_closed
    assert app.state.suno.http.is_closed
