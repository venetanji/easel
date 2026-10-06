"""Video presets, resource admission and durable output metadata; no GPU requests."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel.video_jobs import make_video_id, parse_video_id, video_timing
from easel.video_graph import VIDEO_SIZES
from easel.video_sizing import VideoLimits, VideoSizingError, parse_video_size
from tests.test_h3_api import H3Comfy


@pytest.mark.parametrize("model", ["ltx-2.5", "minimax-h3"])
@pytest.mark.parametrize("size", ["512x512", "640x640", "768x768", "1024x1024", "1024x576"])
def test_shared_square_and_widescreen_dimensions(model, size):
    assert parse_video_size(size, model=model) == tuple(int(value) for value in size.split("x"))


@pytest.mark.parametrize("size", ["1280x720", "720x1280"])
@pytest.mark.parametrize("model", ["ltx-2.5", "minimax-h3"])
def test_legacy_720p_aliases_resolve_to_actual_output_dimensions(model, size):
    assert VIDEO_SIZES[size] == ((1280, 704) if size == "1280x720" else (704, 1280))
    assert parse_video_size(size, model=model) == ((1280, 704) if size == "1280x720" else (704, 1280))


@pytest.mark.parametrize("size", ["auto", "square", "1000x1000", "0x512", "256x0", "999999x256", "-256x512", "1536x1536", "2048x2048"])
def test_bad_or_oversized_dimensions_are_rejected(size):
    with pytest.raises(VideoSizingError):
        parse_video_size(size)


def test_custom_dimensions_follow_model_grid_without_rounding():
    assert parse_video_size(" 1152X640 ") == (1152, 640)
    assert parse_video_size("864x480", model="minimax-h3") == (864, 480)
    with pytest.raises(VideoSizingError, match="multiples of 64"):
        parse_video_size("864x480")


def test_pixel_and_workload_limits_are_inclusive():
    assert parse_video_size("2048x1024") == (2048, 1024)
    limits = VideoLimits(max_pixels=512 * 512, max_pixel_frames=512 * 512 * 25)
    limits.check_frames("ltx-2.5", 512, 512, 25)
    with pytest.raises(VideoSizingError):
        limits.check_frames("ltx-2.5", 512, 512, 49)
    with pytest.raises(VideoSizingError):
        parse_video_size("576x512", limits=limits)


@pytest.mark.parametrize("model", ["ltx-2.5", "minimax-h3"])
def test_every_advertised_preset_accepts_exact_maximum_and_rejects_next_step(model):
    limits = VideoLimits()
    discovery = limits.discovery(model)
    assert "640x640" in discovery["sizes"]
    for entry in discovery["size_presets"]:
        width, height = parse_video_size(entry["size"], model=model, limits=limits)
        assert (width, height) == (entry["width"], entry["height"])
        maximum = entry["max_frames"]
        limits.check_frames(model, width, height, maximum)
        step = 24 if model == "ltx-2.5" else 17
        with pytest.raises(VideoSizingError) as failure:
            limits.check_frames(model, width, height, maximum + step)
        assert failure.value.code == "video_resource_limit"
        assert failure.value.details["max_frames"] == maximum
        assert entry["validation"] == "admission_only"
    assert discovery["size_constraints"]["oom_guarantee"] is False


@pytest.mark.parametrize("values", [(0, 100), (100, -1), (True, 100)])
def test_invalid_limits_fail_startup(values):
    with pytest.raises(ValueError):
        VideoLimits(*values)


def test_configured_budgets_are_parsed_and_validated():
    settings = Settings.from_env({"EASEL_VIDEO_MAX_PIXELS": "1048576", "EASEL_VIDEO_MAX_PIXEL_FRAMES": "70000000"})
    assert settings.video_max_pixels == 1048576
    assert settings.video_max_pixel_frames == 70000000
    with pytest.raises(ValueError):
        Settings.from_env({"EASEL_VIDEO_MAX_PIXELS": "0"})


@pytest.fixture
def api(tmp_path):
    backend = H3Comfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url="http://video", image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        yield client, backend


@pytest.mark.parametrize("model", ["ltx-2.5", "minimax-h3"])
def test_square_dimensions_reach_graph_and_survive_retrieval(api, model):
    client, backend = api
    receipt = client.post("/v1/videos", data={"model": model, "prompt": "square test", "size": "640x640"})
    assert receipt.status_code == 200
    body = receipt.json()
    assert body["size"] == "640x640"
    kind = "EmptyLTXVLatentVideo" if model == "ltx-2.5" else "MiniMaxH3ImageToVideo"
    dimensions = next(node["inputs"] for node in backend.submitted_graph.values() if node["class_type"] == kind)
    expected = 320 if model == "ltx-2.5" else 640
    assert dimensions["width"] == dimensions["height"] == expected
    assert client.get("/v1/videos/" + body["id"]).json()["size"] == "640x640"
    assert client.get("/v1/videos/queue/" + body["id"]).json()["frames"] == body["frames"]


@pytest.mark.parametrize("model, fields", [
    ("ltx-2.5", {"size": "1920x1088", "seconds": "4"}),
    ("ltx-2.5", {"size": "1280x720", "seconds": "12"}),
    ("minimax-h3", {"size": "1024x1024", "frames": "158"}),
    ("minimax-h3", {"size": "1792x1024"}),
])
def test_resolution_duration_budget_rejected_before_any_backend_work(api, model, fields):
    client, backend = api
    response = client.post("/v1/videos", data={"model": model, "prompt": "too large", **fields})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "video_resource_limit"
    assert error["param"] == ("seconds" if model == "ltx-2.5" else "frames")
    assert not backend.info_calls and not backend.uploaded
    assert backend.queue_calls == 0 and backend.submitted_graph is None


def test_short_high_resolution_video_is_admitted(api):
    client, backend = api
    response = client.post("/v1/videos", data={"model": "ltx-2.5", "prompt": "high resolution", "size": "1920x1088", "seconds": "3"})
    assert response.status_code == 200
    assert response.json()["size"] == "1920x1088"
    assert backend.submitted_graph is not None


def test_deployment_limits_are_consistent_between_discovery_and_admission(tmp_path):
    backend = H3Comfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url="http://video", image_job_dir=str(tmp_path),
                                   video_max_pixels=512 * 512, video_max_pixel_frames=512 * 512 * 25)
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        capabilities = client.get("/v1/videos/capabilities").json()
        assert "1024x1024" not in capabilities["sizes"]
        preset = next(entry for entry in capabilities["size_presets"] if entry["size"] == "512x512")
        assert preset["max_seconds"] == 1
        response = client.post("/v1/videos", data={"model": "ltx-2.5", "prompt": "over budget", "size": "512x512", "seconds": "2"})
        assert response.status_code == 400
        assert backend.submitted_graph is None


@pytest.mark.parametrize("model, frames, size", [("ltx-2.5", 97, (1280, 704)), ("minimax-h3", 141, (640, 640))])
def test_new_video_ids_preserve_timing_and_dimensions(model, frames, size):
    video_id = make_video_id("sizing-prompt-id", model, frames=frames, size=size)
    assert parse_video_id(video_id)[1:] == (model, "sizing-prompt-id")
    assert video_timing(video_id) == {"frames": frames, "fps": 24, "duration": frames / 24,
                                     "size": f"{size[0]}x{size[1]}"}


def test_legacy_video_ids_remain_readable():
    assert video_timing(make_video_id("legacy-prompt-id", "ltx-2.5")) == {}
    assert video_timing(make_video_id("legacy-prompt-id", "minimax-h3", frames=124)) == {
        "frames": 124, "fps": 24, "duration": 124 / 24,
    }


@pytest.mark.parametrize("metadata", [":97:640x641", ":96:640x640", ":97:640x640:extra", ":97:999999x640", ":nan:640x640"])
def test_malformed_video_metadata_is_rejected(metadata):
    legacy = make_video_id("legacy-prompt-id", "ltx-2.5")
    parts = legacy.split("_", 3)
    parts[2] += metadata
    with pytest.raises(ValueError):
        parse_video_id("_".join(parts))
