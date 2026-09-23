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
    latents = _nodes(graph, "EmptyFlux2LatentImage") or _nodes(graph, "EmptyLatentImage")
    return latents[0]["inputs"]["batch_size"]


def _output_count(graph):
    """Mirror comfy: a prompt batcher fans out to one image per line; else batch_size."""
    batchers = _nodes(graph, "SimplePromptBatcher")
    if batchers:
        return len([ln for ln in batchers[0]["inputs"]["prompts"].split("\n") if ln])
    repeated_latents = _nodes(graph, "RepeatLatentBatch")
    if repeated_latents:
        return repeated_latents[0]["inputs"]["amount"]
    latent_nodes = _nodes(graph, "EmptyFlux2LatentImage") or _nodes(graph, "EmptyLatentImage")
    return latent_nodes[0]["inputs"]["batch_size"] if latent_nodes else 1


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
        n = _output_count(self.submitted_graph)
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
    assert ids == {"flux2-9b", "flux2-4b", "qwen-image-2.1"}
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


def test_generations_separator_prompt_fans_out():
    client, fake = build()
    r = client.post("/v1/images/generations",
                    json={"model": "flux2-9b", "prompt": "a red fox|||a blue fox|||a green fox"})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    batcher = _nodes(fake.submitted_graph, "SimplePromptBatcher")[0]
    assert [ln for ln in batcher["inputs"]["prompts"].split("\n") if ln] == \
        ["a red fox", "a blue fox", "a green fox"]


def test_generations_normal_prompt_no_batcher():
    client, fake = build()
    client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "line one\nline two"})
    # newlines are NOT the separator; a normal multi-line prompt stays a single prompt
    assert not _nodes(fake.submitted_graph, "SimplePromptBatcher")


def test_edits_separator_prompt_fans_out():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "flux2-9b", "prompt": "make it red|||make it night"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2
    assert _nodes(fake.submitted_graph, "SimplePromptBatcher")
    # reference still encoded once
    assert len(fake.uploaded) == 1


def test_fanout_capped_at_16():
    client, _ = build()
    parts = "|||".join(f"p{i}" for i in range(17))
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": parts})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "prompt"


def test_fanout_16_ok():
    client, fake = build()
    parts = "|||".join(f"p{i}" for i in range(16))
    r = client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": parts})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 16


def test_variations_separator_in_server_prompt_fans_out():
    client, fake = build(variation_prompt="front view|||side view|||back view")
    r = client.post("/v1/images/variations",
                    data={"model": "flux2-9b"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    assert _nodes(fake.submitted_graph, "SimplePromptBatcher")


def test_generations_9b_uses_matching_clip():
    client, fake = build()
    client.post("/v1/images/generations", json={"model": "flux2-9b", "prompt": "p"})
    assert _nodes(fake.submitted_graph, "UNETLoader")[0]["inputs"]["unet_name"] == "flux-2-klein-9b-fp8.safetensors"
    assert _nodes(fake.submitted_graph, "CLIPLoader")[0]["inputs"]["clip_name"] == "qwen_3_8b_fp8mixed.safetensors"


def test_generations_qwen_image_21_uses_standard_graph_and_template_defaults():
    client, fake = build()
    r = client.post("/v1/images/generations", json={
        "model": "qwen-image-2.1", "prompt": "a red fox", "size": "768x512", "seed": 42,
    })
    assert r.status_code == 200
    graph = fake.submitted_graph
    assert _nodes(graph, "UNETLoader")[0]["inputs"]["unet_name"] == \
        "qwen_image_2.1_int8_convrot.safetensors"
    assert _nodes(graph, "CLIPLoader")[0]["inputs"] == {
        "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
        "type": "qwen_image",
        "device": "default",
    }
    assert _nodes(graph, "VAELoader")[0]["inputs"]["vae_name"] == \
        "qwen_image_2.1_vae_bf16.safetensors"
    text = _nodes(graph, "TextEncodeQwenImage21")[0]
    assert text["inputs"]["prompt"] == "a red fox"
    assert text["inputs"]["negative_prompt"] == ""
    sampler = _nodes(graph, "KSampler")[0]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"],
            sampler["scheduler"], sampler["seed"]) == (25, 1, "euler", "simple", 42)
    assert _nodes(graph, "EmptyLatentImage")[0]["inputs"] == {
        "width": 768, "height": 512, "batch_size": 1,
    }
    assert not _nodes(graph, "Flux2Scheduler")


def test_generations_qwen_image_21_honors_steps_and_prompt_fanout():
    client, fake = build()
    r = client.post("/v1/images/generations", json={
        "model": "qwen-image-2.1", "prompt": "a fox|||a wolf", "steps": 6,
    })
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2
    batcher = _nodes(fake.submitted_graph, "SimplePromptBatcher")[0]
    assert batcher["inputs"]["prompts"] == "a fox\na wolf\n"
    assert _nodes(fake.submitted_graph, "KSampler")[0]["inputs"]["steps"] == 6


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


def test_qwen_image_21_edits_use_encoder_reference_inputs_and_latent():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "qwen-image-2.1", "prompt": "combine them"},
                    files=[("image[]", ("a.png", b"A", "image/png")),
                           ("image[]", ("b.png", b"B", "image/png"))])
    assert r.status_code == 200
    graph = fake.submitted_graph
    encoder = _nodes(graph, "TextEncodeQwenImage21")[0]
    load_ids = [key for key, node in graph.items() if node["class_type"] == "LoadImage"]
    assert encoder["inputs"]["images.image_1"] == [load_ids[0], 0]
    assert encoder["inputs"]["images.image_2"] == [load_ids[1], 0]
    assert encoder["inputs"]["resolution"] == 1024
    assert len(_nodes(graph, "LoadImage")) == 2
    encoder_id = next(k for k, v in graph.items() if v["class_type"] == "TextEncodeQwenImage21")
    sampler = _nodes(graph, "KSampler")[0]["inputs"]
    assert sampler["latent_image"] == [encoder_id, 2]
    assert sampler["model"][0] == next(
        k for k, v in graph.items() if v["class_type"] == "QwenImage21Cache"
    )
    assert len(fake.uploaded) == 2


def test_qwen_image_21_variations_use_reference_encoder():
    client, fake = build(variation_prompt="front view")
    r = client.post("/v1/images/variations", data={"model": "qwen-image-2.1"},
                    files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert _nodes(fake.submitted_graph, "TextEncodeQwenImage21")
    assert _nodes(fake.submitted_graph, "LoadImage")


def test_qwen_image_21_edits_reject_more_than_16_reference_images():
    client, fake = build()
    r = client.post("/v1/images/edits",
                    data={"model": "qwen-image-2.1", "prompt": "combine them"},
                    files=[("image[]", (f"{i}.png", b"A", "image/png")) for i in range(17)])
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "image"
    assert fake.uploaded == []


def test_qwen_image_21_edit_n_repeats_encoded_latent():
    client, fake = build()
    r = client.post("/v1/images/edits", data={
        "model": "qwen-image-2.1", "prompt": "make it brighter", "n": "3",
    }, files={"image": ("a.png", b"A", "image/png")})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    repeated = _nodes(fake.submitted_graph, "RepeatLatentBatch")
    assert len(repeated) == 1 and repeated[0]["inputs"]["amount"] == 3


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


def test_view_requires_auth_when_key_set():
    client, _ = build(api_key="secret")
    r = client.get("/v1/images/view", params={"filename": "out_0.png", "type": "output"})
    assert r.status_code == 401
    ok = client.get("/v1/images/view", params={"filename": "out_0.png", "type": "output"},
                    headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


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
