"""Route-level tests for the FastAPI app, against a fake comfy client (no network)."""
import base64
import dataclasses

import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel.comfy_client import ComfySubmitError, ComfyExecError, ComfyTimeout


# ---- fakes / helpers ----

def _nodes(graph, class_type):
    return [n for n in graph.values() if n["class_type"] == class_type]


def _batch_size(graph):
    return _nodes(graph, "EmptyFlux2LatentImage")[0]["inputs"]["batch_size"]


class FakeComfy:
    def __init__(self, *, images=None, submit_exc=None, wait_exc=None):
        self.images = images
        self.submit_exc = submit_exc
        self.wait_exc = wait_exc
        self.submitted_graph = None
        self.uploaded = []
        self.prompt_id = "fake-pid"

    async def upload_image(self, data, filename):
        self.uploaded.append((data, filename))
        return f"srv_{len(self.uploaded)}.png", ""

    async def submit(self, graph):
        self.submitted_graph = graph
        if self.submit_exc:
            raise self.submit_exc
        return self.prompt_id

    async def wait(self, prompt_id, timeout, poll_interval=0.5):
        if self.wait_exc:
            raise self.wait_exc
        if self.images is not None:
            return self.images
        n = _batch_size(self.submitted_graph)
        return [{"filename": f"out_{i}.png", "subfolder": "", "type": "output"} for i in range(n)]

    async def fetch(self, ref):
        return b"PNG:" + ref["filename"].encode()


def settings(**over):
    return dataclasses.replace(Settings.from_env({}), **over)


def build(fake=None, **over):
    fake = fake or FakeComfy()
    app = create_app(settings=settings(**over), comfy=fake)
    return TestClient(app), fake


# ---- health & models ----

def test_health_open_no_auth():
    client, _ = build(api_key="secret")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_list_shape():
    client, _ = build()
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"flux2-9b", "flux2-4b"}
    for m in body["data"]:
        assert m["object"] == "model" and "created" in m and "owned_by" in m


# ---- generations ----

def test_generations_b64_happy_path():
    client, fake = build()
    r = client.post("/v1/images/generations", json={"model": "flux2-4b", "prompt": "a red fox"})
    assert r.status_code == 200
    body = r.json()
    assert "created" in body and len(body["data"]) == 1
    raw = base64.b64decode(body["data"][0]["b64_json"])
    assert raw.startswith(b"PNG:")
    # correct UNET + matching CLIP + prompt wired into the graph
    assert _nodes(fake.submitted_graph, "UNETLoader")[0]["inputs"]["unet_name"] == "flux-2-klein-4b-fp8.safetensors"
    assert _nodes(fake.submitted_graph, "CLIPLoader")[0]["inputs"]["clip_name"] == "qwen_3_4b_fp4_flux2.safetensors"
    texts = [n["inputs"]["text"] for n in _nodes(fake.submitted_graph, "CLIPTextEncode")]
    assert "a red fox" in texts


def test_generations_9b_uses_matching_clip():
    client, fake = build()
    client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert _nodes(fake.submitted_graph, "UNETLoader")[0]["inputs"]["unet_name"] == "flux-2-klein-9b-fp8.safetensors"
    assert _nodes(fake.submitted_graph, "CLIPLoader")[0]["inputs"]["clip_name"] == "qwen_3_8b_fp8mixed.safetensors"


def test_generations_url_format():
    client, _ = build()
    r = client.post("/v1/images/generations",
                    json={"model": "flux2-9b", "prompt": "p", "response_format": "url"})
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert "/v1/images/view" in url and "filename=out_0.png" in url


def test_generations_n_maps_to_batch_size():
    client, fake = build()
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p", "n": 3})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    assert _batch_size(fake.submitted_graph) == 3


def test_generations_size_parsed_into_dimensions():
    client, fake = build()
    r = client.post("/v1/images/generations",
                    json={"model": "flux2-9b", "prompt": "p", "size": "512x768"})
    assert r.status_code == 200
    latent = _nodes(fake.submitted_graph, "EmptyFlux2LatentImage")[0]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (512, 768)


def test_generations_unknown_model_400():
    client, _ = build()
    r = client.post("/v1/images/generations", json={"model": "dall-e-3", "prompt": "p"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "model_not_found"
    assert err["param"] == "model"


def test_generations_missing_prompt_400():
    client, _ = build()
    r = client.post("/v1/images/generations", json={"model": "flux2-9b"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "prompt"


def test_generations_bad_size_400():
    client, _ = build()
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p", "size": "banana"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "size"


def test_generations_ignores_unknown_sdk_fields():
    client, _ = build()
    r = client.post("/v1/images/generations", json={
        "model": "flux2-9b", "prompt": "p",
        "user": "u1", "quality": "hd", "style": "vivid", "background": "opaque",
    })
    assert r.status_code == 200


def test_generations_bad_n_400():
    client, _ = build()
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p", "n": 99})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"


# ---- edits ----

def test_edits_happy_path_uploads_and_builds_reference_graph():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b", "prompt": "make it snowy"},
                    files={"image": ("photo.png", b"IMGDATA", "image/png")})
    assert r.status_code == 200
    assert len(fake.uploaded) == 1 and fake.uploaded[0][0] == b"IMGDATA"
    # reference-edit graph: exactly one CLIPTextEncode + ConditioningZeroOut + ReferenceLatent
    assert len(_nodes(fake.submitted_graph, "ReferenceLatent")) == 2  # 1 ref x pos+neg
    assert _nodes(fake.submitted_graph, "ConditioningZeroOut")
    # uploaded server filename is the one wired into LoadImage
    assert _nodes(fake.submitted_graph, "LoadImage")[0]["inputs"]["image"] == "srv_1.png"


def test_edits_multiple_reference_images():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b", "prompt": "combine them"},
                    files=[("image[]", ("a.png", b"A", "image/png")),
                           ("image[]", ("b.png", b"B", "image/png"))])
    assert r.status_code == 200
    assert len(fake.uploaded) == 2
    assert len(_nodes(fake.submitted_graph, "ReferenceLatent")) == 4  # 2 refs x pos+neg


def test_edits_mask_rejected_400():
    client, _ = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b", "prompt": "p"},
                    files=[("image", ("a.png", b"A", "image/png")),
                           ("mask", ("m.png", b"M", "image/png"))])
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "mask"


def test_edits_missing_image_400():
    client, _ = build()
    r = client.post("/v1/images/edits", data={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "image"


def test_edits_missing_prompt_400():
    client, _ = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "prompt"


def test_edits_explicit_size_overrides_derivation():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b", "prompt": "p", "size": "512x512"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    latent = _nodes(fake.submitted_graph, "EmptyFlux2LatentImage")[0]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (512, 512)
    assert not _nodes(fake.submitted_graph, "GetImageSize")


def test_edits_default_size_derives_from_image():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b", "prompt": "p"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert _nodes(fake.submitted_graph, "GetImageSize")


# ---- variations ----

def test_variations_injects_default_prompt_single_ref():
    client, fake = build(variation_prompt="vary this")
    r = client.post("/v1/images/variations",
                    data={"model": "flux2-9b"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert len(fake.uploaded) == 1
    # single-ref reference edit using the configured variation prompt
    assert len(_nodes(fake.submitted_graph, "ReferenceLatent")) == 2
    assert _nodes(fake.submitted_graph, "CLIPTextEncode")[0]["inputs"]["text"] == "vary this"


def test_variations_missing_image_400():
    client, _ = build()
    r = client.post("/v1/images/variations", data={"model": "flux2-9b"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "image"


def test_variations_n_maps_to_batch_size():
    client, fake = build()
    r = client.post("/v1/images/variations",
                    data={"model": "flux2-9b", "n": "2"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert _batch_size(fake.submitted_graph) == 2


# ---- view proxy ----

def test_view_proxy_returns_image_bytes():
    client, _ = build()
    r = client.get("/v1/images/view", params={"filename": "out_0.png", "subfolder": "", "type": "output"})
    assert r.status_code == 200
    assert r.content == b"PNG:out_0.png"
    assert r.headers["content-type"].startswith("image/")


# ---- auth ----

def test_auth_required_when_key_set():
    client, _ = build(api_key="secret")
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_auth_accepts_correct_bearer():
    client, _ = build(api_key="secret")
    r = client.post("/v1/images/generations",
                    headers={"Authorization": "Bearer secret"},
                    json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 200


def test_auth_rejects_wrong_bearer():
    client, _ = build(api_key="secret")
    r = client.post("/v1/images/generations",
                    headers={"Authorization": "Bearer nope"},
                    json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 401


# ---- concurrency & upstream errors ----

def test_busy_returns_429_with_retry_after():
    client, _ = build(max_inflight=1)
    # occupy the only slot
    assert client.app.state.inflight.acquire() is True
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 429
    assert "retry-after" in {k.lower() for k in r.headers}
    assert r.json()["error"]["type"] == "rate_limit_error"


def test_submit_error_maps_to_502_with_node_errors():
    fake = FakeComfy(submit_exc=ComfySubmitError({"3": {"errors": [{"message": "bad"}]}}))
    client, _ = build(fake)
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 502
    assert "node_errors" in r.json()["error"]["message"]


def test_exec_error_maps_to_502():
    fake = FakeComfy(wait_exc=ComfyExecError("pid-1", "CUDA OOM", "KSampler"))
    client, _ = build(fake)
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 502
    assert "CUDA OOM" in r.json()["error"]["message"]


def test_timeout_maps_to_504_with_prompt_id():
    fake = FakeComfy(wait_exc=ComfyTimeout("pid-42"))
    client, _ = build(fake)
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert r.status_code == 504
    assert "pid-42" in r.json()["error"]["message"]


def test_inflight_released_after_request():
    client, _ = build(max_inflight=1)
    client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    # slot should be free again
    assert client.app.state.inflight.acquire() is True
