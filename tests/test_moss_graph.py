import pytest

from easel.moss_graph import MODEL_PATH, text_to_image
from tests.test_app import FakeComfy, build


def inputs(graph, kind):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == kind)


def test_native_moss_workflow_matches_upstream_sampling_and_ports():
    graph = text_to_image(prompt="A lighthouse", seed=0)
    assert len(graph) == 9
    assert inputs(graph, "MOSSImageHFLoader") == {
        "model_path": MODEL_PATH, "weight_dtype": "fp8_e4m3fn", "text_device": "cpu",
    }
    assert graph["2"]["inputs"] == {"text": "A lighthouse", "clip": ["1", 1]}
    assert graph["3"]["inputs"] == {"text": "", "clip": ["1", 1]}
    assert inputs(graph, "ModelSamplingSD3") == {"model": ["1", 0], "shift": 5.0}
    assert inputs(graph, "SkipLayerGuidanceDiTSimple") == {
        "model": ["5", 0], "double_layers": "", "single_layers": "22",
        "start_percent": 0.0, "end_percent": 1.0,
    }
    assert inputs(graph, "KSampler") == {
        "model": ["6", 0], "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": ["4", 0], "seed": 0, "steps": 50, "cfg": 3.0,
        "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1.0,
    }
    assert inputs(graph, "VAEDecode") == {"samples": ["7", 0], "vae": ["1", 2]}


def test_generation_routes_to_image_with_model_defaults():
    video = FakeComfy()
    client, image = build(video=video, comfy_video_url="http://video")
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-base", "prompt": "A lighthouse", "seed": 0, "size": "512x768",
    })
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert inputs(image.submitted_graph, "KSampler")["steps"] == 50
    assert inputs(image.submitted_graph, "KSampler")["seed"] == 0
    assert inputs(image.submitted_graph, "EmptyFlux2LatentImage") == {
        "width": 512, "height": 768, "batch_size": 1,
    }
    assert video.submitted_graph is None


def test_moss_model_path_and_steps_override():
    client, image = build(moss_model_path="/other/model")
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-base", "prompt": "fixture", "steps": 2,
    })
    assert response.status_code == 200
    assert inputs(image.submitted_graph, "MOSSImageHFLoader")["model_path"] == "/other/model"
    assert inputs(image.submitted_graph, "KSampler")["steps"] == 2


def test_moss_cannot_be_routed_to_video():
    video = FakeComfy()
    client, image = build(video=video, comfy_video_url="http://video")
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-base", "prompt": "fixture", "server": "video",
    })
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "server"
    assert video.submitted_graph is None and image.submitted_graph is None


@pytest.mark.parametrize("endpoint", ["edits", "variations"])
def test_unsupported_reference_operations_fail_before_upload(endpoint):
    client, image = build()
    response = client.post("/v1/images/" + endpoint,
                           data={"model": "moss-image-base", "prompt": "fixture"},
                           files={"image": ("ref.png", b"PNG", "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_operation"
    assert image.uploaded == [] and image.submitted_graph is None


@pytest.mark.parametrize("prompt, count", [("fixture", 3), ("one ||| two", 2)])
def test_moss_supports_existing_batch_and_prompt_fanout(prompt, count):
    client, image = build()
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-base", "prompt": prompt, "n": 3,
    })
    assert response.status_code == 200
    assert len(response.json()["data"]) == count
    if "|||" in prompt:
        assert inputs(image.submitted_graph, "SimplePromptBatcher")["prompts"] == "one\ntwo\n"


def test_model_path_environment_configuration():
    from easel.config import Settings

    assert Settings.from_env({"MOSS_IMAGE_MODEL_PATH": " /custom/model "}).moss_model_path == "/custom/model"
    assert Settings.from_env({"MOSS_IMAGE_MODEL_PATH": " "}).moss_model_path == MODEL_PATH


def test_moss_async_admission_keeps_model_id_and_image_backend(tmp_path):
    from fastapi.testclient import TestClient
    from tests.test_image_jobs import QueuedComfy
    from tests.test_model_serving import serving_app

    image, video = QueuedComfy(), QueuedComfy()
    client = TestClient(serving_app(tmp_path, image, video))
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-base", "prompt": "fixture", "seed": 42,
    }, headers={"Prefer": "respond-async"})
    assert response.status_code == 202
    receipt = response.json()
    assert receipt["model"] == "moss-image-base"
    assert len(image.submissions) == 1 and video.submissions == []
    assert inputs(image.submissions[0], "KSampler")["steps"] == 50


def test_openapi_only_offers_moss_for_generation():
    from tests.test_openapi import request_schema

    client, _ = build()
    schema = client.get("/openapi.json").json()
    generation = request_schema(schema, "/v1/images/generations", "application/json")
    assert "moss-image-base" in generation["properties"]["model"]["enum"]
    for endpoint in ("edits", "variations"):
        form = request_schema(schema, "/v1/images/" + endpoint, "multipart/form-data")
        assert "moss-image-base" not in form["properties"]["model"]["enum"]
