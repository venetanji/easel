"""Integration tests against a real ComfyUI Flux.2 server.

Opt-in: set EASEL_INTEGRATION=1 and COMFY_URL_FLUX to a reachable server.
Uses the 4b UNET + low steps to keep it fast.
"""
import dataclasses
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from easel.app import create_app
from easel.config import Settings

pytestmark = pytest.mark.integration

if os.environ.get("EASEL_INTEGRATION") != "1":
    pytest.skip("set EASEL_INTEGRATION=1 to run integration tests", allow_module_level=True)

MODEL = "flux2-4b"
SIZE = "512x512"
STEPS = 4


def _settings():
    return dataclasses.replace(
        Settings.from_env({}),
        comfy_url=os.environ.get("COMFY_URL_FLUX", "http://10.99.0.7:8188"),
        job_timeout=300,
        max_inflight=1,
    )


@pytest.fixture()
def client():
    with TestClient(create_app(settings=_settings())) as c:
        yield c


def _red_png(size=(512, 512)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _decode(b64: str) -> Image.Image:
    import base64
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img


def test_text_to_image(client):
    r = client.post("/v1/images/generations",
                    json={"model": MODEL, "prompt": "a small red apple on a white table",
                          "size": SIZE, "steps": STEPS})
    assert r.status_code == 200, r.text
    img = _decode(r.json()["data"][0]["b64_json"])
    assert img.format == "PNG"
    assert img.size == (512, 512)


def test_image_edit(client):
    r = client.post("/v1/images/edits",
                    data={"model": MODEL, "prompt": "change the background to deep blue",
                          "size": SIZE, "steps": str(STEPS)},
                    files={"image": ("red.png", _red_png(), "image/png")})
    assert r.status_code == 200, r.text
    img = _decode(r.json()["data"][0]["b64_json"])
    assert img.format == "PNG"
    assert img.size == (512, 512)


def test_image_variation(client):
    r = client.post("/v1/images/variations",
                    data={"model": MODEL, "size": SIZE, "steps": str(STEPS)},
                    files={"image": ("red.png", _red_png(), "image/png")})
    assert r.status_code == 200, r.text
    img = _decode(r.json()["data"][0]["b64_json"])
    assert img.format == "PNG"
    assert img.size == (512, 512)
