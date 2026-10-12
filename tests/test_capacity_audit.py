"""The real-capacity runner preserves receipts and never retries generation."""
import argparse
import json

import httpx
import pytest

from scripts import audit_capacity


def options(tmp_path, **overrides):
    return argparse.Namespace(url="http://easel", output=str(tmp_path / "report.json"),
        live=True, resume=False, submit_only=False, images=1, videos=0, audio=0,
        image_size="512x512", image_steps=25, video_size="512x320", video_seconds=1,
        poll_seconds=10, **overrides)


def mock_client(monkeypatch, handler):
    client = httpx.AsyncClient(base_url="http://easel", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(audit_capacity.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(audit_capacity, "load_dotenv", lambda: None)
    return client


async def test_live_submission_is_opt_in(tmp_path):
    settings = options(tmp_path)
    settings.live = False
    with pytest.raises(SystemExit, match="--live"):
        await audit_capacity.audit(settings)
    assert not (tmp_path / "report.json").exists()


async def test_receipt_is_saved_before_polling_and_output_is_checked(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        saved = json.loads((tmp_path / "report.json").read_text())
        if request.method == "POST":
            assert saved["requests"][0]["submission_uncertain"] is True
            return httpx.Response(202, json={"id": "image-existing", "status": "queued"})
        assert saved["requests"][0]["receipt"]["id"] == "image-existing"
        if request.url.path.endswith("/content/0"):
            return httpx.Response(200, content=b"real-output", headers={"content-type": "image/png"})
        return httpx.Response(200, json={"status": "completed"})

    mock_client(monkeypatch, handler)
    await audit_capacity.audit(options(tmp_path))
    saved = json.loads((tmp_path / "report.json").read_text())
    assert len([request for request in requests if request.method == "POST"]) == 1
    assert saved["requests"][0]["output"]["bytes"] == 11
    assert saved["unfinished_gpu_jobs"] == []
    assert (tmp_path / "report.json").stat().st_mode & 0o777 == 0o600


async def test_ambiguous_transport_failure_is_recorded_without_retry(tmp_path, monkeypatch):
    submissions = []

    def handler(request):
        if request.method == "POST":
            submissions.append(request)
            raise httpx.ReadTimeout("receipt lost", request=request)
        return httpx.Response(200, json={"status": "ok"})

    mock_client(monkeypatch, handler)
    await audit_capacity.audit(options(tmp_path))
    item = json.loads((tmp_path / "report.json").read_text())["requests"][0]
    assert len(submissions) == 1
    assert item["submission_uncertain"] is True
    assert item["error"] == "ReadTimeout"


@pytest.mark.parametrize("response", [
    httpx.Response(504, json={"error": {"code": "upstream_timeout"}}),
    httpx.Response(200, json={"status": "unknown"}),
    httpx.Response(200, json=[]),
])
async def test_missing_receipts_or_server_timeout_remain_uncertain(tmp_path, monkeypatch, response):
    submissions = []

    def handler(request):
        if request.method == "POST":
            submissions.append(request)
            return response
        return httpx.Response(200, json={"status": "ok"})

    mock_client(monkeypatch, handler)
    await audit_capacity.audit(options(tmp_path))
    item = json.loads((tmp_path / "report.json").read_text())["requests"][0]
    assert len(submissions) == 1
    assert item["submission_uncertain"] is True


async def test_resume_uses_saved_ids_without_any_post_or_unconfigured_audio(tmp_path, monkeypatch):
    audit_capacity.save_report(tmp_path / "report.json", {
        "base_url": "http://easel", "health": [], "requests": [{
            "kind": "video", "index": 0, "http_status": 200, "receipt": {"id": "existing-video"}}]})

    def handler(request):
        assert request.method == "GET"
        assert "/audio/" not in request.url.path
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"video", headers={"content-type": "video/mp4"})
        return httpx.Response(200, json={"status": "completed"})

    mock_client(monkeypatch, handler)
    settings = options(tmp_path)
    settings.resume, settings.audio = True, 2
    await audit_capacity.audit(settings)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["requests"][0]["final"]["status"] == "completed"


async def test_unresolved_suno_attempt_prevents_new_generations(tmp_path, monkeypatch):
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"status": "captcha_required", "attempt_id": "existing-attempt"})

    mock_client(monkeypatch, handler)
    settings = options(tmp_path)
    settings.audio = 2
    with pytest.raises(SystemExit, match="unresolved"):
        await audit_capacity.audit(settings)
    assert json.loads((tmp_path / "report.json").read_text())["requests"] == []
