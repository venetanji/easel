"""MOSS plugin recipe, image routing and durable receipt contracts."""
import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easel.app import build_generation_graph, create_app
from easel.config import Settings, resolve_model
from tests.test_app import _nodes, build
from tests.test_image_jobs import QueuedComfy


def expanded(graph, node_id):
    node = graph[node_id]
    return {"class_type": node["class_type"], "inputs": {
        key: expanded(graph, value[0]) if isinstance(value, list) and len(value) == 2
        and isinstance(value[0], str) and value[0] in graph else value
        for key, value in node["inputs"].items()}}


def test_base_recipe_matches_independent_pinned_upstream_graph():
    spec = resolve_model("moss-image-base")
    reference = json.loads((Path(__file__).parent / "fixtures/moss/plugin-api.json").read_text())
    reference["1"]["inputs"].update(model_path=spec.hf_model_path,
                                     weight_dtype="fp8_e4m3fn", text_device="cpu")
    reference["4"]["inputs"]["text"] = "A mossy stone golem."
    reference["18"]["inputs"]["filename_prefix"] = "easel_moss_t2i"
    graph = build_generation_graph(spec, prompt="A mossy stone golem.", width=1024,
                                   height=1024, steps=spec.default_steps, batch_size=1, seed=42)
    save_id = next(nid for nid, node in graph.items() if node["class_type"] == "SaveImage")
    assert len(graph) == len(reference)
    assert expanded(graph, save_id) == expanded(reference, "18")
    assert not spec.supports_edits
    assert spec.default_steps == 50


def test_moss_generation_routes_to_hf_loader_with_model_defaults():
    client, comfy = build(default_steps=8)
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-base", "prompt": "A mossy stone golem.",
        "size": "512x768", "seed": 0, "n": 2,
    })
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    graph = comfy.submitted_graph
    loader = _nodes(graph, "MOSSImageHFLoader")[0]["inputs"]
    assert loader == {"model_path": resolve_model("moss-image-base").hf_model_path,
                      "weight_dtype": "fp8_e4m3fn", "text_device": "cpu"}
    assert _nodes(graph, "EmptyFlux2LatentImage")[0]["inputs"] == {
        "width": 512, "height": 768, "batch_size": 2}
    sampler = _nodes(graph, "KSampler")[0]["inputs"]
    assert sampler["seed"] == 0 and sampler["steps"] == 50
    assert not _nodes(graph, "TextEncodeQwenImage21")


@pytest.mark.parametrize("model", ["moss-image-base", "moss-image-sft-8step"])
def test_moss_fanout_and_step_override_use_common_image_contract(model):
    client, comfy = build()
    response = client.post("/v1/images/generations", json={
        "model": model, "prompt": "A fox|||A wolf", "n": 4, "steps": 20,
    })
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    graph = comfy.submitted_graph
    assert _nodes(graph, "KSampler")[0]["inputs"]["steps"] == 20
    assert _nodes(graph, "SimplePromptBatcher")[0]["inputs"]["prompts"] == "A fox\nA wolf\n"
    assert _nodes(graph, "EmptyFlux2LatentImage")[0]["inputs"]["batch_size"] == 1


@pytest.mark.parametrize("endpoint", ["edits", "variations"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model", ["moss-image-base", "moss-image-sft-8step"])
def test_moss_rejects_reference_operations_before_upload_or_admission(tmp_path, endpoint, asynchronous, model):
    comfy = QueuedComfy()
    settings = dataclasses.replace(Settings.from_env({}), image_job_dir=str(tmp_path))
    client = TestClient(create_app(settings=settings, comfy=comfy))
    response = client.post("/v1/images/" + endpoint, data={
        "model": model, "prompt": "Make it snowy",
    }, files={"image": ("ref.png", b"REFERENCE", "image/png")},
        headers={"Prefer": "respond-async"} if asynchronous else {})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_operation"
    assert response.json()["error"]["param"] == "model"
    assert comfy.uploaded == comfy.submissions == []
    assert client.app.state.image_jobs.store.active() == []


@pytest.mark.parametrize("model", ["moss-image-base", "moss-image-sft-8step"])
def test_moss_async_receipt_caches_output_and_recovers_without_resubmission(tmp_path, model):
    comfy = QueuedComfy()
    settings = dataclasses.replace(Settings.from_env({}), image_job_dir=str(tmp_path))
    app = create_app(settings=settings, comfy=comfy)
    client = TestClient(app)
    response = client.post("/v1/images/generations", json={
        "model": model, "prompt": "A mossy stone golem.",
        "size": "512x512", "steps": 20, "seed": 42, "response_format": "url",
    }, headers={"Prefer": "respond-async"})
    assert response.status_code == 202
    receipt = response.json()
    assert receipt["model"] == model
    assert receipt["status"] == "queued"
    comfy.finish("image-prompt-1")
    recovered = TestClient(create_app(settings=settings, comfy=comfy))
    completed = recovered.get(response.headers["Location"]).json()
    assert completed["status"] == "completed"
    assert recovered.get(completed["data"][0]["url"]).content == b"PNG:out.png"
    assert len(comfy.submissions) == 1


def test_dmd_profile_uses_matching_sft_bf16_and_unguided_eight_step_recipe():
    client, comfy = build(default_steps=25)
    response = client.post("/v1/images/generations", json={
        "model": "moss-image-sft-8step", "prompt": "A mossy stone golem.", "seed": 0,
    })
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    graph = comfy.submitted_graph
    loader_id, loader = next((nid, node["inputs"]) for nid, node in graph.items()
                             if node["class_type"] == "MOSSImageHFLoader")
    assert loader == {"model_path": "/app/ComfyUI/models/moss-image/MOSS-Image-SFT",
                      "weight_dtype": "bf16", "text_device": "cpu"}
    lora_id, lora = next((nid, node["inputs"]) for nid, node in graph.items()
                         if node["class_type"] == "LoraLoaderModelOnly")
    assert lora == {"model": [loader_id, 0],
                    "lora_name": "moss_image_sft_8steps_dmd.safetensors", "strength_model": 1.0}
    shifted_id, shifted = next((nid, node["inputs"]) for nid, node in graph.items()
                               if node["class_type"] == "ModelSamplingSD3")
    assert shifted == {"model": [lora_id, 0], "shift": 5}
    sampler = _nodes(graph, "KSampler")[0]["inputs"]
    assert sampler["model"] == [shifted_id, 0]
    assert sampler["steps"] == 8 and sampler["cfg"] == 1 and sampler["seed"] == 0
    assert sampler["sampler_name"] == "euler" and sampler["scheduler"] == "sgm_uniform"
    assert _nodes(graph, "EmptyFlux2LatentImage")[0]["inputs"] == {
        "width": 1024, "height": 1024, "batch_size": 1}
    assert not _nodes(graph, "SkipLayerGuidanceDiTSimple")
    assert not resolve_model("moss-image-sft-8step").supports_edits
