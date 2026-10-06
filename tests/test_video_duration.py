"""Video duration admission and frame counts against a fake ComfyUI client."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings


class FakeVideoComfy:
    def __init__(self):
        self.submitted_graph = None
        self.uploaded = []
        self.queue_calls = 0

    async def queue(self):
        self.queue_calls += 1
        return {"queue_running": [], "queue_pending": []}

    async def submit(self, graph):
        self.submitted_graph = graph
        return "duration-test-prompt-id"

    async def upload_image(self, data, filename, content_type):
        self.uploaded.append((data, filename, content_type))
        return "reference.png", ""


@pytest.fixture
def video_api():
    video = FakeVideoComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url="http://video")
    app = create_app(settings=settings, comfy=FakeVideoComfy(), comfy_video=video)
    with TestClient(app) as client:
        yield client, video


def _create_video(client, seconds=None, with_reference=False):
    data = {"model": "ltx-2.5", "prompt": "a red ball rolling on a table", "size": "512x320"}
    if seconds is not None:
        data["seconds"] = str(seconds)
    files = None
    if with_reference:
        files = {"input_reference": ("reference.png", b"PNGREFERENCE", "image/png")}
    return client.post("/v1/videos", data=data, files=files)


@pytest.mark.parametrize("seconds", range(1, 13))
@pytest.mark.parametrize("with_reference", [False, True])
def test_video_duration_accepts_every_whole_second(video_api, seconds, with_reference):
    client, video = video_api
    response = _create_video(client, seconds, with_reference)
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    expected_frames = seconds * 24 + 1
    video_latent = next(node for node in video.submitted_graph.values()
                        if node["class_type"] == "EmptyLTXVLatentVideo")
    audio_latent = next(node for node in video.submitted_graph.values()
                        if node["class_type"] == "LTXVEmptyLatentAudio")
    assert video_latent["inputs"]["length"] == expected_frames
    assert audio_latent["inputs"]["frames_number"] == expected_frames
    assert expected_frames % 8 == 1
    assert len(video.uploaded) == int(with_reference)


@pytest.mark.parametrize("with_reference", [False, True])
def test_video_duration_default_stays_four_seconds(video_api, with_reference):
    client, video = video_api
    response = _create_video(client, with_reference=with_reference)
    assert response.status_code == 200
    latent = next(node for node in video.submitted_graph.values()
                  if node["class_type"] == "EmptyLTXVLatentVideo")
    assert latent["inputs"]["length"] == 97


@pytest.mark.parametrize("seconds", [-1, 0, 13, 100])
@pytest.mark.parametrize("with_reference", [False, True])
def test_video_duration_rejects_outside_range(video_api, seconds, with_reference):
    client, video = video_api
    response = _create_video(client, seconds, with_reference)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "seconds"
    assert response.json()["error"]["message"] == "seconds must be between 1 and 12"
    assert video.queue_calls == 0
    assert video.submitted_graph is None
    assert video.uploaded == []


@pytest.mark.parametrize("seconds", ["0.5", "1.5", "12.5", "not-a-number"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_video_duration_rejects_fractional_or_invalid_values(video_api, seconds,
                                                           with_reference):
    client, video = video_api
    response = _create_video(client, seconds, with_reference)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "seconds"
    assert video.queue_calls == 0
    assert video.submitted_graph is None
    assert video.uploaded == []
