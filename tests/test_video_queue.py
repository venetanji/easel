"""Video queue vocabulary, completion guards and runtime estimates."""
import dataclasses
import time

from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel.video_jobs import VIDEO_TTL_SECONDS, make_video_id
from tests.test_image_jobs import QueuedComfy


def build(tmp_path):
    video = QueuedComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url="http://video",
                                   image_job_dir=str(tmp_path))
    app = create_app(settings=settings, comfy=video, comfy_video=video)
    return TestClient(app), video


def test_video_queue_exposes_retention_and_lifecycle_fields(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("video-prompt-1", "ltx-2.5")
    video.pending = [[1, "video-prompt-1"]]
    response = client.get("/v1/videos/queue/" + video_id)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["expires_at"] - body["created_at"] == VIDEO_TTL_SECONDS
    assert body["queue_position"] == 1
    assert body["estimated_wait_seconds"] is None
    assert body["error"] is None
    assert body["completed_at"] is None


def test_partial_video_outputs_are_not_terminal(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("video-prompt-1", "ltx-2.5")
    video.histories["video-prompt-1"] = {
        "status": {"status_str": "running", "completed": False},
        "outputs": {"preview": {"videos": [{"filename": "preview.mp4"}]}},
    }
    assert client.get("/v1/videos/" + video_id).json()["status"] == "in_progress"


def test_running_estimate_decreases_from_observed_start(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("video-prompt-1", "ltx-2.5")
    video.running = [[1, "video-prompt-1"]]
    client.app.state.video_runtimes["known"] = 30
    client.app.state.video_started_at[video_id] = int(time.time()) - 10
    body = client.get("/v1/videos/queue/" + video_id).json()
    assert body["queue_position"] == 0
    assert 19 <= body["estimated_wait_seconds"] <= 20
    assert body["estimated_completion_at"] >= int(time.time()) + 19


def test_missing_execution_timestamp_does_not_count_queue_time_as_runtime(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("video-prompt-1", "ltx-2.5", int(time.time()) - 1000)
    video.histories["video-prompt-1"] = {
        "status": {"status_str": "success", "completed": True},
        "outputs": {"save": {"videos": [{"filename": "out.mp4"}]}},
    }
    body = client.get("/v1/videos/queue/" + video_id).json()
    assert body["status"] == "completed"
    assert body["average_generation_seconds"] is None
    assert body["estimated_wait_seconds"] == 0


def test_interrupted_video_reports_cancelled_without_resubmission(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("video-prompt-1", "ltx-2.5")
    video.finish("video-prompt-1", status="error", outputs=False, event="execution_interrupted")
    body = client.get("/v1/videos/queue/" + video_id).json()
    assert body["status"] == "cancelled"
    assert body["error"]["code"] == "upstream_cancelled"
    assert body["estimated_completion_at"] is None
    assert video.submissions == []


def test_terminal_failure_ignores_stale_running_queue_snapshot(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("video-prompt-1", "ltx-2.5")
    video.finish("video-prompt-1", status="error", outputs=False)
    video.running = [[1, "video-prompt-1"]]
    client.app.state.video_runtimes["known"] = 30
    body = client.get("/v1/videos/queue/" + video_id).json()
    assert body["status"] == "failed"
    assert body["queue_position"] is None
    assert body["estimated_wait_seconds"] is None
    assert body["estimated_completion_at"] is None
