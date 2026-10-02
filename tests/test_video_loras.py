"""Adapter admission, discovery, two-pass wiring, and speed conditioning."""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel.video_loras import VIDEO_LORAS, installed_lora_names, parse_video_loras
from tests.test_video_duration import FakeVideoComfy


class LoRAComfy(FakeVideoComfy):
    def __init__(self):
        super().__init__()
        self.installed = [asset["filename"] for entry in VIDEO_LORAS.values()
                          for asset in entry["files"]]
        self.info_calls = 0
        self.missing_nodes = set()

    async def object_info(self, class_name):
        self.info_calls += 1
        if class_name in self.missing_nodes:
            return {}
        if class_name != "LoraLoaderModelOnly":
            return {class_name: {"input": {}}}
        return {class_name: {"input": {"required": {"lora_name": [self.installed]}}}}


@pytest.fixture
def lora_api():
    video = LoRAComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url="http://video")
    app = create_app(settings=settings, comfy=LoRAComfy(), comfy_video=video)
    with TestClient(app) as client:
        yield client, video


def create(client, with_reference=False, lora_sheet=False, **options):
    data = {"model": "ltx-2.5", "prompt": "a red ball on a wooden table",
            "seconds": "2", "size": "512x320", **options}
    files = None
    if with_reference:
        files = {"input_reference": ("reference.png", b"PNGREFERENCE", "image/png")}
    if lora_sheet:
        files = {**(files or {}), "lora_reference": ("sheet.png", b"PNGREFERENCE", "image/png")}
    return client.post("/v1/videos", data=data, files=files)


@pytest.mark.parametrize("camera", ["dolly-in", "dolly-out", "dolly-left", "dolly-right",
                                    "jib-up", "jib-down", "static"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_all_camera_adapters_feed_both_sampling_passes(lora_api, camera, with_reference):
    client, video = lora_api
    response = create(client, with_reference, camera_lora=camera, seed="0")
    assert response.status_code == 200
    loader_id, loader = next((node_id, node) for node_id, node in video.submitted_graph.items()
                             if node["class_type"] == "LoraLoaderModelOnly")
    assert loader["inputs"]["lora_name"] == VIDEO_LORAS["camera-" + camera]["files"][0]["filename"]
    assert loader["inputs"]["strength_model"] == 0.8
    guiders = [node for node in video.submitted_graph.values()
               if node["class_type"] == "LTXVDualCFGGuider"]
    assert len(guiders) == 2
    assert all(node["inputs"]["model"] == [loader_id, 0] for node in guiders)
    assert [node["inputs"]["noise_seed"] for node in video.submitted_graph.values()
            if node["class_type"] == "RandomNoise"] == [0, 1]


def test_loras_stack_in_order(lora_api):
    client, video = lora_api
    response = create(client, camera_lora="dolly_in", loras=json.dumps([
        {"id": "camera-jib-up", "strength": 0.4}]))
    assert response.status_code == 200
    loaders = [(node_id, node) for node_id, node in video.submitted_graph.items()
               if node["class_type"] == "LoraLoaderModelOnly"]
    assert len(loaders) == 2
    assert loaders[1][1]["inputs"]["model"] == [loaders[0][0], 0]
    assert loaders[1][1]["inputs"]["strength_model"] == 0.4


@pytest.mark.parametrize("options", [
    {"camera_lora": "unknown"}, {"camera_lora_strength": "0.5"},
    {"camera_lora": "static", "camera_lora_strength": "nan"},
    {"camera_lora": "static", "camera_lora_strength": "-1"},
    {"camera_lora": "static", "camera_lora_strength": "2.1"},
    {"loras": "invalid"}, {"loras": "{}"}, {"loras": '["camera-static"]'},
    {"loras": '[{"id":"../../secret.safetensors"}]'},
    {"loras": '[{"id":"camera-static","strength":true}]'},
    {"loras": '[{"id":"camera-static","strength":NaN}]'},
    {"loras": '[{"id":"camera-static","strength":"1"}]'},
    {"loras": json.dumps([{"id": "camera-static", "strength": 10 ** 400}])},
    {"loras": '[{"id":"camera-static","filename":"secret"}]'},
    {"loras": '[{"id":"day-to-night"}]'},
    {"camera_lora": "static", "loras": '[{"id":"camera-static"}]'},
    {"motion_speed": "0.2"}, {"seed": "-1"}, {"seed": str(2 ** 64 - 1)},
    {"loras": "x" * 8193},
    {"loras": "[" * 1100 + "]" * 1100},
    {"loras": json.dumps([{"id":"camera-static"}] * 5)},
])
def test_invalid_adapter_options_fail_before_upstream_work(lora_api, options):
    client, video = lora_api
    response = create(client, True, **options)
    assert response.status_code == 400
    assert video.info_calls == video.queue_calls == 0
    assert video.uploaded == []
    assert video.submitted_graph is None


def test_missing_weights_fail_before_queue_and_upload(lora_api):
    client, video = lora_api
    video.installed = []
    response = create(client, True, camera_lora="dolly-in")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "lora_not_installed"
    assert video.queue_calls == 0
    assert video.uploaded == []
    assert video.submitted_graph is None


def test_catalog_is_static_route_and_marks_missing_and_specialized_adapters(lora_api):
    client, video = lora_api
    video.installed = [VIDEO_LORAS["camera-static"]["files"][0]["filename"]]
    response = client.get("/v1/videos/loras")
    assert response.status_code == 200
    entries = {entry["id"]: entry for entry in response.json()["data"]}
    assert len(entries) == len(VIDEO_LORAS)
    assert entries["camera-static"]["installed"] is True
    assert entries["camera-dolly-in"]["installed"] is False
    assert entries["ingredients"]["supported"] is True
    assert entries["ingredients"]["requires"] == ["lora_reference"]
    assert entries["day-to-night"]["supported"] is False


def test_catalog_requires_auth(lora_api):
    client, video = lora_api
    protected = create_app(settings=dataclasses.replace(Settings.from_env({}),
        comfy_video_url="http://video", api_key="secret"), comfy=video, comfy_video=video)
    with TestClient(protected) as protected_client:
        assert protected_client.get("/v1/videos/loras").status_code == 401
        assert protected_client.get("/v1/videos/loras", headers={
            "Authorization": "Bearer secret"}).status_code == 200


@pytest.mark.parametrize("model_id", ["cinemagraph", "slow-motion"])
def test_image_adapters_require_a_reference(lora_api, model_id):
    client, video = lora_api
    response = create(client, loras=json.dumps([{"id": model_id}]),
                      **({"motion_speed": "0.2"} if model_id == "slow-motion" else {}))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "input_reference"
    assert video.info_calls == video.queue_calls == 0


def test_cinemagraph_adds_its_trigger(lora_api):
    client, video = lora_api
    assert create(client, True, loras='[{"id":"cinemagraph"}]').status_code == 200
    texts = [node["inputs"]["text"] for node in video.submitted_graph.values()
             if node["class_type"] == "CLIPTextEncode"]
    assert texts[0].startswith("CINEMAGRAPH_MOTION, ")


def test_cinemagraph_rejects_moving_camera_stack(lora_api):
    client, video = lora_api
    response = create(client, True, camera_lora="dolly-in", loras='[{"id":"cinemagraph"}]')
    assert response.status_code == 400
    assert video.info_calls == video.queue_calls == 0


def test_slow_motion_keeps_playback_and_audio_fps_independent(lora_api):
    client, video = lora_api
    response = create(client, True, loras='[{"id":"slow-motion"}]', motion_speed="0.2")
    assert response.status_code == 200
    for node in video.submitted_graph.values():
        if node["class_type"] == "LTXVConditioning":
            assert node["inputs"]["frame_rate"] == 120
        if node["class_type"] == "LTXVEmptyLatentAudio":
            assert node["inputs"]["frame_rate"] == 24
        if node["class_type"] == "CreateVideo":
            assert node["inputs"]["fps"] == 24


@pytest.mark.parametrize("speed", [None, "0", "0.01", "1.1", "nan", "inf"])
def test_slow_motion_rejects_invalid_or_missing_speed(lora_api, speed):
    client, video = lora_api
    options = {"motion_speed": speed} if speed is not None else {}
    response = create(client, True, loras='[{"id":"slow-motion"}]', **options)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "motion_speed"
    assert video.info_calls == video.queue_calls == 0


def test_backend_dynamic_combo_schema_is_supported():
    assert installed_lora_names({"LoraLoaderModelOnly": {"input": {"required": {
        "lora_name": ["COMBO", {"options": ["adapter.safetensors"]}]}}}}) == {"adapter.safetensors"}


def test_maximum_combined_adapter_count():
    requested = json.dumps([{"id": "camera-" + camera} for camera in (
        "dolly-in", "dolly-out", "jib-up", "static")])
    with pytest.raises(ValueError, match="at most 4"):
        parse_video_loras(requested, "dolly-left", None)


def test_ingredients_has_conditioning_and_cropping_in_both_passes(lora_api):
    client, video = lora_api
    response = create(client, lora_sheet=True, seconds="5", loras='[{"id":"ingredients"}]',
                      lora_reference_strength="0.5")
    assert response.status_code == 200
    graph = video.submitted_graph
    loaders = [(node_id, node) for node_id, node in graph.items()
               if node["class_type"] == "LTXICLoRALoaderModelOnly"]
    assert len(loaders) == 1
    guides = [(node_id, node) for node_id, node in graph.items()
              if node["class_type"] == "LTXAddVideoICLoRAGuide"]
    assert len(guides) == 2
    assert all(node["inputs"]["latent_downscale_factor"] == [loaders[0][0], 1]
               and node["inputs"]["strength"] == 0.5 for _node_id, node in guides)
    crops = [(node_id, node) for node_id, node in graph.items()
             if node["class_type"] == "LTXVCropGuides"]
    assert len(crops) == 2
    upsampler = next(node for node in graph.values() if node["class_type"] == "LTXVLatentUpsampler")
    decoder = next(node for node in graph.values() if node["class_type"] == "VAEDecodeTiled")
    assert upsampler["inputs"]["samples"] == [crops[0][0], 2]
    assert decoder["inputs"]["samples"] == [crops[1][0], 2]
    assert next(node for node in graph.values()
                if node["class_type"] == "RepeatImageBatch")["inputs"]["amount"] == 121
    assert len(video.uploaded) == 1


@pytest.mark.parametrize("options", [
    {"seconds": "4", "lora_sheet": True}, {"seconds": "5"},
    {"seconds": "5", "lora_sheet": True, "camera_lora": "static"},
    {"seconds": "5", "lora_sheet": True, "lora_reference_strength": "nan"},
    {"seconds": "5", "lora_sheet": True, "lora_reference_strength": "1.1"},
])
def test_ingredients_admission_is_fail_closed(lora_api, options):
    client, video = lora_api
    response = create(client, loras='[{"id":"ingredients"}]', **options)
    assert response.status_code == 400
    assert video.info_calls == video.queue_calls == 0
    assert video.uploaded == []


def test_orphan_lora_reference_is_rejected(lora_api):
    client, video = lora_api
    assert create(client, lora_sheet=True).status_code == 400
    assert video.info_calls == video.queue_calls == 0


def test_missing_ic_nodes_fail_before_upload(lora_api):
    client, video = lora_api
    video.missing_nodes = {"LTXAddVideoICLoRAGuide"}
    response = create(client, lora_sheet=True, seconds="5", loras='[{"id":"ingredients"}]')
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ic_lora_nodes_unavailable"
    assert video.queue_calls == 0
    assert video.uploaded == []
