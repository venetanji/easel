"""Video queue vocabulary, completion guards and runtime estimates."""
import dataclasses
import time

import pytest

from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel.video_jobs import VIDEO_TTL_SECONDS, make_video_id, parse_video_id
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


@pytest.mark.parametrize("exception_type, message", [
    ("torch.OutOfMemoryError", "allocation failed"),
    ("torch.cuda.OutOfMemoryError", ""),
    ("RuntimeError", "CUDA out of memory. Tried to allocate 1 GiB"),
    ("RuntimeError", "Allocation on device"),
    ("RuntimeError", "CUDA OOM"),
])
def test_oom_is_actionable_and_does_not_resubmit_or_block_other_jobs(tmp_path, exception_type, message):
    client, video = build(tmp_path)
    failed_id = make_video_id("oom-prompt-id", "ltx-2.5", frames=97, size=(1024, 1024))
    video.histories["oom-prompt-id"] = {
        "status": {"status_str": "error", "messages": [["execution_error", {
            "exception_type": exception_type, "exception_message": message,
        }]]}, "outputs": {},
    }
    for route in ("/v1/videos/", "/v1/videos/queue/"):
        failed = client.get(route + failed_id).json()
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "upstream_out_of_memory"
        assert failed["error"]["retryable"] is False
        assert failed["error"]["suggested_action"] == "reduce_size_or_duration"
    assert video.submissions == []
    receipt = client.post("/v1/videos", data={"model": "ltx-2.5", "prompt": "explicit smaller recovery job",
                                           "size": "512x512", "seconds": "1"})
    assert receipt.status_code == 200 and len(video.submissions) == 1
    recovered_id = receipt.json()["id"]
    prompt_id = parse_video_id(recovered_id)[2]
    video.finish(prompt_id)
    video.histories[prompt_id]["outputs"] = {"save": {"videos": [{"filename": "recovered.mp4"}]}}
    assert client.get("/v1/videos/" + recovered_id).json()["status"] == "completed"


def test_non_memory_failure_keeps_generic_error_code(tmp_path):
    client, video = build(tmp_path)
    video_id = make_video_id("invalid-prompt-id", "ltx-2.5")
    video.histories["invalid-prompt-id"] = {
        "status": {"status_str": "error", "messages": [["execution_error", {
            "exception_type": "RuntimeError", "exception_message": "tensor dimensions do not match",
        }]]}, "outputs": {},
    }
    body = client.get("/v1/videos/" + video_id).json()
    assert body["status"] == "failed" and body["error"]["code"] == "upstream_execution_error"
