"""The live benchmark plans safely, persists receipts and never replaces jobs."""
import json

import httpx
import pytest

from easel.video_sizing import VideoLimits
from scripts import benchmark_video as benchmark


def capabilities(model="ltx-2.5"):
    return {"model": model, "fps": 24, "seconds": {"min": 1, "max": 12},
            "frames": {"min": 124, "max": 362, "step": 17, "offset": 5},
            **VideoLimits().discovery(model)}


def report(cases=None):
    return {"model": "ltx-2.5", "prompt": "benchmark", "url": "http://easel",
            "cases": [benchmark.plan_case(value, capabilities()) for value in (cases or ["512x512:1"])]}


def test_plan_obeys_published_aliases_and_model_temporal_grids():
    assert benchmark.plan_case("1280x720:4", capabilities())["size"] == "1280x704"
    assert benchmark.plan_case("640x640:141", capabilities("minimax-h3"))["expected_frames"] == 141


@pytest.mark.parametrize("value", ["1920x1088:4", "1024x1024:7", "1000x1000:1", "512x512:0", "512x512:13", "garbage"])
def test_invalid_or_over_budget_plan_is_rejected(value):
    with pytest.raises(ValueError):
        benchmark.plan_case(value, capabilities())


def test_h3_plan_rejects_invalid_temporal_grid():
    with pytest.raises(ValueError):
        benchmark.plan_case("640x640:125", capabilities("minimax-h3"))


def test_dry_run_does_not_submit_or_create_receipt_directory(tmp_path, monkeypatch, capsys):
    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1/videos/capabilities"
        return httpx.Response(200, json=capabilities())
    client = httpx.Client(base_url="http://easel", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(benchmark.httpx, "Client", lambda **kwargs: client)
    output = tmp_path / "unused"
    assert benchmark.main(["--url", "http://easel", "--case", "640x640:1", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert not output.exists()


def test_jobs_are_serial_and_receipts_are_saved_before_first_poll(tmp_path, monkeypatch):
    data = report(["512x512:1", "640x640:1"])
    submissions = []
    def handler(request):
        saved = json.loads((tmp_path / "report.json").read_text())
        if request.method == "POST":
            index = len(submissions)
            assert saved["cases"][index]["status"] == "submitting"
            assert all(case["status"] == "verified" for case in saved["cases"][:index])
            submissions.append(f"video-{index}")
            return httpx.Response(200, json={"id": submissions[-1]})
        assert saved["cases"][len(submissions) - 1]["receipt"]["id"] == submissions[-1]
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"video")
        return httpx.Response(200, json={"status": "completed"})
    monkeypatch.setattr(benchmark, "verify_video", lambda path, case: {"full_decode": "passed"})
    with httpx.Client(base_url="http://easel", transport=httpx.MockTransport(handler)) as client:
        assert benchmark.run_matrix(client, tmp_path, data, timeout=10, poll_interval=0) == 0
    assert len(submissions) == 2
    assert all(case["status"] == "verified" for case in data["cases"])
    assert (tmp_path / "report.json").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "case-01.mp4").stat().st_mode & 0o777 == 0o600


def test_timeout_resume_polls_existing_receipt_without_new_generation(tmp_path, monkeypatch):
    data = report()
    submissions = []
    status = {"value": "queued"}
    def handler(request):
        if request.method == "POST":
            submissions.append(request)
            return httpx.Response(200, json={"id": "existing-video-id"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"video")
        return httpx.Response(200, json={"status": status["value"]})
    monkeypatch.setattr(benchmark, "verify_video", lambda path, case: {"full_decode": "passed"})
    with httpx.Client(base_url="http://easel", transport=httpx.MockTransport(handler)) as client:
        assert benchmark.run_matrix(client, tmp_path, data, timeout=-1, poll_interval=0) == 1
        resumed = json.loads((tmp_path / "report.json").read_text())
        status["value"] = "completed"
        assert benchmark.run_matrix(client, tmp_path, resumed, timeout=10, poll_interval=0) == 0
    assert len(submissions) == 1


def test_oom_stops_matrix_without_retry_or_submitting_next_case(tmp_path):
    data = report(["512x512:1", "640x640:1"])
    submissions = []
    def handler(request):
        if request.method == "POST":
            submissions.append(request)
            return httpx.Response(200, json={"id": "oom-video-id"})
        return httpx.Response(200, json={"status": "failed", "error": {"code": "upstream_out_of_memory"}})
    with httpx.Client(base_url="http://easel", transport=httpx.MockTransport(handler)) as client:
        assert benchmark.run_matrix(client, tmp_path, data, timeout=10, poll_interval=0) == 1
        assert benchmark.run_matrix(client, tmp_path, data, timeout=10, poll_interval=0) == 1
    assert len(submissions) == 1 and data["cases"][1]["status"] == "planned"


def test_lost_receipt_blocks_replacement_submission_on_resume(tmp_path):
    data = report()
    submissions = []
    def handler(request):
        assert request.method == "POST"
        submissions.append(request)
        raise httpx.ReadError("response lost", request=request)
    with httpx.Client(base_url="http://easel", transport=httpx.MockTransport(handler)) as client:
        assert benchmark.run_matrix(client, tmp_path, data, timeout=10, poll_interval=0) == 1
        saved = json.loads((tmp_path / "report.json").read_text())
        assert benchmark.run_matrix(client, tmp_path, saved, timeout=10, poll_interval=0) == 1
    assert len(submissions) == 1


def test_unknown_status_stops_with_receipt_preserved(tmp_path):
    data = report()
    submissions = []
    def handler(request):
        if request.method == "POST":
            submissions.append(request)
            return httpx.Response(200, json={"id": "existing-video-id"})
        return httpx.Response(200, json={"status": "unknown"})
    with httpx.Client(base_url="http://easel", transport=httpx.MockTransport(handler)) as client:
        assert benchmark.run_matrix(client, tmp_path, data, timeout=10, poll_interval=0) == 1
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["cases"][0]["receipt"]["id"] == "existing-video-id"
    assert len(submissions) == 1
