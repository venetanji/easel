"""Opt-in deployed Suno tests; real generation additionally requires explicit opt-in."""
import json
import os
import time
from pathlib import Path

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("EASEL_AUDIO_INTEGRATION") != "1",
                       reason="set EASEL_AUDIO_INTEGRATION=1 to test deployed Suno routes"),
]


@pytest.fixture
def deployed_audio():
    headers = {}
    if os.environ.get("EASEL_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["EASEL_API_KEY"]
    with httpx.Client(base_url=os.environ.get("EASEL_INTEGRATION_URL", "http://127.0.0.1:8799"),
                      headers=headers, timeout=180) as client:
        yield client


@pytest.mark.parametrize("model,prompt", [
    ("suno-music", "Gentle ambient piano with warm strings"),
    ("suno-speech", "This is Easel's audio routing verification."),
    ("suno-sound", "A single soft wooden chime, no voices or music"),
    ("suno-music", "Delicate acoustic guitar with soft percussion"),
])
def test_deployed_audio_dry_run(deployed_audio, model, prompt):
    discovery = deployed_audio.get("/v1/models")
    discovery.raise_for_status()
    assert model in {entry["id"] for entry in discovery.json()["data"]}
    before = deployed_audio.get("/v1/audio/generations/status")
    before.raise_for_status()
    if before.json()["status"] in {"pending", "captcha_required"}:
        pytest.skip("an existing Suno submission needs confirmation; do not interfere")
    response = deployed_audio.post("/v1/audio/generations", json={
        "model": model, "prompt": prompt, "dry_run": True,
    })
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "prepared"
    assert any(field.get("value", "").strip() == prompt for field in response.json()["fields"])
    after = deployed_audio.get("/v1/audio/generations/status")
    after.raise_for_status()
    assert after.json().get("metrics", {}).get("attempts") == before.json().get("metrics", {}).get("attempts")


@pytest.mark.skipif(os.environ.get("EASEL_AUDIO_GENERATION") != "1",
                    reason="set EASEL_AUDIO_GENERATION=1 to spend credits on one sound generation")
def test_deployed_sound_generation_and_download(deployed_audio, tmp_path):
    before = deployed_audio.get("/v1/audio/generations/status")
    before.raise_for_status()
    if before.json()["status"] in {"pending", "captcha_required"}:
        pytest.skip("an existing Suno submission needs confirmation; do not submit again")
    response = deployed_audio.post("/v1/audio/generations", json={
        "model": "suno-sound", "prompt": "A single soft wooden chime, no voices or music",
    })
    assert response.status_code in (200, 202), response.text
    result = response.json()
    deadline = time.monotonic() + 300
    while result["status"] in {"pending", "captcha_required"}:
        if result["status"] == "captcha_required":
            pytest.skip("Solve CAPTCHA manually, then poll status without resubmitting: " + result["novnc_url"])
        assert time.monotonic() < deadline, "submission remains ambiguous; inspect status before retrying"
        time.sleep(5)
        response = deployed_audio.get(result["status_url"])
        response.raise_for_status()
        result = response.json()
    assert result["status"] in {"submitted", "complete"}, result
    assert result["songs"], result
    track = result["songs"][0]
    while track["status"] != "complete":
        assert track["status"] != "error", track
        assert time.monotonic() < deadline, "track still generating; inspect its ID without resubmitting"
        time.sleep(5)
        response = deployed_audio.get("/v1/audio/tracks/" + track["id"])
        response.raise_for_status()
        track = response.json()
    saved = deployed_audio.post(track["download_url"])
    assert saved.status_code == 200, saved.text
    metadata = saved.json()
    audio = deployed_audio.get(metadata["audio_url"])
    assert audio.status_code == 200, audio.text[:300]
    assert audio.content and len(audio.content) == metadata["size"]
    assert audio.headers["content-type"].startswith(("audio/", "video/"))
    directory = Path(os.environ.get("EASEL_AUDIO_ARTIFACT_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / metadata["filename"]).write_bytes(audio.content)
    (directory / "summary.json").write_text(json.dumps({
        "generation": result, "track": track, "download": metadata,
    }, indent=2) + "\n")
