"""Integration tests against a real ComfyUI Qwen Image 2.1 server.

Opt-in: set EASEL_INTEGRATION=1 and COMFY_URL_IMAGE to a reachable server.
Set EASEL_INTEGRATION_URL to test a deployed Easel instead of an in-process app;
set EASEL_API_KEY if that deployment requires authentication.
Uses an explicit low step count to bound protocol smoke tests.
"""
import dataclasses
import io
import os

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from easel.app import create_app
from easel.config import Settings

pytestmark = pytest.mark.integration

if os.environ.get("EASEL_INTEGRATION") != "1":
    pytest.skip("set EASEL_INTEGRATION=1 to run integration tests", allow_module_level=True)

MODEL = "qwen-image-2.1"
SIZE = "512x512"
STEPS = 4


def _settings():
    return dataclasses.replace(
        Settings.from_env({}),
        comfy_url=os.environ.get("COMFY_URL_IMAGE") or os.environ.get("COMFY_URL_FLUX", "http://10.99.0.7:8188"),
        job_timeout=300,
        max_inflight=1,
    )


@pytest.fixture()
def client():
    base_url = os.environ.get("EASEL_INTEGRATION_URL")
    if base_url:
        api_key = os.environ.get("EASEL_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with httpx.Client(base_url=base_url, headers=headers, timeout=600) as deployed:
            yield deployed
        return
    with TestClient(create_app(settings=_settings())) as c:
        yield c


def _red_png(size=(512, 512)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _transparent_png(mode="RGBA") -> bytes:
    image = Image.new("RGBA", (512, 512), (10, 70, 140, 0))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((112, 304, 400, 432), fill=(200, 30, 30, 128))
    drawing.ellipse((128, 96, 384, 352), fill=(200, 30, 30, 255))
    if mode == "P":
        image = image.quantize(colors=8, dither=Image.Dither.NONE)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    assert image.convert("RGBA").getchannel("A").getextrema() == (0, 255)
    return buffer.getvalue()


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


def test_multiprompt_generation_fans_out(client):
    r = client.post("/v1/images/generations",
                    json={"model": MODEL, "size": SIZE, "steps": STEPS,
                          "prompt": "a red apple|||a green pear|||a yellow banana"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data) == 3
    for item in data:
        assert _decode(item["b64_json"]).size == (512, 512)


def test_edit_without_size_derives_from_image(client):
    # Qwen's reference encoder derives an aligned canvas from the input aspect ratio.
    r = client.post("/v1/images/edits",
                    data={"model": MODEL, "prompt": "add soft morning light", "steps": str(STEPS)},
                    files={"image": ("wide.png", _red_png((768, 512)), "image/png")})
    assert r.status_code == 200, r.text
    w, h = _decode(r.json()["data"][0]["b64_json"]).size
    assert w % 16 == 0 and h % 16 == 0, (w, h)


def test_variation_without_size_derives_from_image(client):
    r = client.post("/v1/images/variations",
                    data={"model": MODEL, "steps": str(STEPS)},
                    files={"image": ("wide.png", _red_png((768, 512)), "image/png")})
    assert r.status_code == 200, r.text
    w, h = _decode(r.json()["data"][0]["b64_json"]).size
    assert w % 16 == 0 and h % 16 == 0, (w, h)


def test_image_variation(client):
    r = client.post("/v1/images/variations",
                    data={"model": MODEL, "size": SIZE, "steps": str(STEPS)},
                    files={"image": ("red.png", _red_png(), "image/png")})
    assert r.status_code == 200, r.text
    img = _decode(r.json()["data"][0]["b64_json"])
    assert img.format == "PNG"
    assert img.size == (512, 512)


@pytest.mark.parametrize("mode", ["RGBA", "P"])
@pytest.mark.parametrize("endpoint", ["edits", "variations"])
def test_transparent_png_reference(client, endpoint, mode):
    response = client.post(
        f"/v1/images/{endpoint}",
        data={"model": MODEL, "prompt": "a red apple, keep the background transparent",
              "size": SIZE, "steps": str(STEPS), "seed": "42"},
        files={"image": (f"transparent-{mode}.png", _transparent_png(mode), "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 1
    image = _decode(data[0]["b64_json"])
    assert image.format == "PNG"
    assert image.mode == "RGBA"
    assert image.size == (512, 512)


def test_transparent_png_multi_reference_batch(client):
    response = client.post(
        "/v1/images/edits",
        data={"model": MODEL, "prompt": "a red apple, keep the background transparent",
              "size": SIZE, "steps": str(STEPS), "seed": "42", "n": "2"},
        files=[("image[]", (f"transparent-{mode}.png", _transparent_png(mode), "image/png"))
               for mode in ("RGBA", "P")],
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 2
    for item in data:
        image = _decode(item["b64_json"])
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (512, 512)
