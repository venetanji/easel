"""Portable H3 recipe parity and temporal-guide graph contracts; no GPU work."""
import json
import re
from pathlib import Path

import pytest

from easel import h3_graph as h3

FIXTURES = Path(__file__).parent / "fixtures" / "h3"
SCHEMAS = json.loads((FIXTURES / "runtime_nodes.json").read_text())["nodes"]


def nodes(graph, kind):
    return [(node_id, node["inputs"]) for node_id, node in graph.items()
            if node["class_type"] == kind]


def expanded(graph, node_id):
    node = graph[node_id]
    return {"class_type": node["class_type"], "inputs": {
        name: {"source": expanded(graph, value[0]), "slot": value[1]}
        if isinstance(value, list) and len(value) == 2 and value[0] in graph else value
        for name, value in node["inputs"].items()
    }}


def validate_schema(graph):
    for node_id, node in graph.items():
        schema = SCHEMAS[node["class_type"]]
        declared = {**schema["input"].get("optional", {}), **schema["input"].get("required", {})}
        assert set(schema["input"].get("required", {})) <= node["inputs"].keys()
        for field, value in node["inputs"].items():
            field_schema = declared.get(field)
            if field_schema is None:
                group, port = field.split(".", 1)
                dynamic = declared[group]
                assert dynamic[0] == "COMFY_AUTOGROW_V3"
                template = dynamic[1]["template"]
                match = re.fullmatch(re.escape(template["prefix"]) + r"(\d+)", port)
                assert match and int(match.group(1)) < template["max"]
                field_schema = next(iter(template["input"]["required"].values()))
            expected = field_schema[0]
            if isinstance(value, list) and len(value) == 2 and value[0] in graph:
                assert int(value[0]) < int(node_id)
                assert SCHEMAS[graph[value[0]]["class_type"]]["output"][value[1]] == expected
            elif isinstance(expected, list) and node["class_type"] != "LoadImage":
                assert value in expected
            elif expected == "COMBO":
                assert value in field_schema[1]["options"]
            elif expected in ("INT", "FLOAT"):
                assert type(value) in ((int,) if expected == "INT" else (int, float))
                bounds = field_schema[1] if len(field_schema) > 1 else {}
                assert bounds.get("min", value) <= value <= bounds.get("max", value)
    assert len(nodes(graph, "SaveVideo")) == 1


@pytest.mark.parametrize("mode", ["t2v", "i2v", "r2v"])
def test_existing_native_recipes_match_independent_prototypes(mode):
    prototype = json.loads((FIXTURES / f"h3_{mode}_prototype.json").read_text())
    native = nodes(prototype, "MiniMaxH3ReferenceToVideo" if mode == "r2v"
                   else "MiniMaxH3ImageToVideo")[0][1]
    options = dict(prompt=native["prompt"], seed=20261002,
                   filename_prefix=nodes(prototype, "SaveVideo")[0][1]["filename_prefix"])
    if mode == "i2v":
        options["image_filename"] = "h3_fixture_first.png"
    if mode == "r2v":
        options["reference_filenames"] = ["h3_fixture_1.png", "h3_fixture_2.jpg"]
    graph = getattr(h3, "build_" + mode)(**options)
    assert len(graph) == len(prototype)
    assert expanded(graph, nodes(graph, "SaveVideo")[0][0]) == expanded(
        prototype, nodes(prototype, "SaveVideo")[0][0])
    validate_schema(graph)


@pytest.mark.parametrize("frames", range(124, 363, 17))
@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_exact_native_frames_and_full_single_pass_seed_range(frames, seed):
    graph = h3.build_t2v("one coherent shot", frames=frames, seed=seed)
    assert nodes(graph, "MiniMaxH3ImageToVideo")[0][1]["length"] == frames
    assert nodes(graph, "RandomNoise")[0][1]["noise_seed"] == seed
    validate_schema(graph)


@pytest.mark.parametrize("options", [
    {"prompt": ""}, {"prompt": " \t"}, {"prompt": "x\x00y"}, {"prompt": True},
    {"frames": 123}, {"frames": 125}, {"frames": 363}, {"frames": True},
    {"frames": 124.0}, {"seed": -1}, {"seed": 2**64}, {"seed": "0"},
    {"seed": True}, {"width": 865}, {"height": 0}, {"width": 16416},
    {"width": True}, {"filename_prefix": "../escape"}, {"filename_prefix": "/absolute"},
    {"filename_prefix": "video/%date%"}, {"filename_prefix": "video/with space"},
    {"filename_prefix": "x" * 201},
])
def test_invalid_shared_controls_are_rejected(options):
    with pytest.raises(ValueError):
        h3.build_t2v(**{"prompt": "test", **options})


@pytest.mark.parametrize("name", ["", "../a.png", "/a.png", "a\\b.png", "a.png [output]",
                                  "a.gif", None, True, "x" * 201 + ".png"])
def test_first_frame_is_a_managed_input_name(name):
    with pytest.raises(ValueError):
        h3.build_i2v("test", image_filename=name)


@pytest.mark.parametrize("references", [[], "a.png", [True], ["../a.png"], ["a.png"] * 3])
def test_semantic_reference_admission(references):
    with pytest.raises(ValueError):
        h3.build_r2v("test", reference_filenames=references)


@pytest.mark.parametrize("size", ["other", None, True])
def test_reference_size_is_explicit(size):
    with pytest.raises(ValueError):
        h3.build_r2v("test", reference_filenames=["a.png"], ref_image_size=size)


@pytest.mark.parametrize("count", [1, 5, 22, 39])
def test_guide_batches_preserve_every_frame_in_order(count):
    filenames = tuple(f"frame_{index:02}.png" for index in range(count))
    graph = h3.build_temporal_guided_video("continue motion", guides=[h3.TemporalGuide(filenames, 0)])
    assert [inputs["image"] for _, inputs in nodes(graph, "LoadImage")] == list(filenames)
    assert len(nodes(graph, "ImageBatch")) == count - 1
    guide = nodes(graph, "MiniMaxH3AddGuide")[0]
    assert guide[1]["frame_idx"] == 0
    assert set(guide[1]) == {"positive", "latent", "vae", "image", "frame_idx"}
    assert nodes(graph, "BasicGuider")[0][1]["conditioning"] == [guide[0], 0]
    assert nodes(graph, "SamplerCustomAdvanced")[0][1]["latent_image"] == guide[1]["latent"]
    assert not nodes(graph, "LTXVCropGuides")
    assert not nodes(graph, "LoraLoaderModelOnly")
    assert not nodes(graph, "MiniMaxH3SigmaShift")
    assert nodes(graph, "KSamplerSelect")[0][1]["sampler_name"] == "res_multistep"
    assert nodes(graph, "BasicScheduler")[0][1]["steps"] == 20
    validate_schema(graph)


@pytest.mark.parametrize("count", [0, 2, 3, 4, 6, 21, 23, 38, 40, 56])
def test_guides_never_silently_truncate(count):
    with pytest.raises(ValueError):
        h3.build_temporal_guided_video("test", guides=[
            h3.TemporalGuide(tuple(f"frame_{index}.png" for index in range(count)), 0)])


def test_guide_chains_sort_target_positions_not_source_frames():
    graph = h3.build_temporal_guided_video("test", guides=[
        h3.TemporalGuide(("later.png",), 123),
        h3.TemporalGuide(tuple(f"frame_{index}.png" for index in range(22)), 0),
    ], reference_filenames=["identity.png"])
    guides = nodes(graph, "MiniMaxH3AddGuide")
    assert [inputs["frame_idx"] for _, inputs in guides] == [0, 123]
    assert guides[1][1]["positive"] == [guides[0][0], 0]
    assert guides[0][1]["latent"] == guides[1][1]["latent"]
    reference = nodes(graph, "MiniMaxH3ReferenceToVideo")[0][1]
    assert reference["ref_images.ref_image_0"] != guides[0][1]["image"]
    validate_schema(graph)


@pytest.mark.parametrize("guides", [
    [], [h3.TemporalGuide(("a.png",), 0)] * 4,
    [h3.TemporalGuide(("a.png",), -1)], [h3.TemporalGuide(("a.png",), 124)],
    [h3.TemporalGuide(("a.png",), True)], [h3.TemporalGuide(("a.png",), 1.0)],
    [h3.TemporalGuide(("../a.png",), 0)], [h3.TemporalGuide(["a.png"], 0)],
    [h3.TemporalGuide(tuple(f"frame_{index}.png" for index in range(22)), 103)],
    [h3.TemporalGuide(("a.png",), 0), h3.TemporalGuide(("b.png",), 0)],
    [h3.TemporalGuide(tuple(f"frame_{index}.png" for index in range(22)), 0),
     h3.TemporalGuide(("b.png",), 21)],
    [{"image_filenames": ["a.png"], "frame_index": 0, "strength": 0.5}],
])
def test_guide_bounds_overlap_types_and_cardinality(guides):
    with pytest.raises(ValueError):
        h3.build_temporal_guided_video("test", guides=guides)


def test_unsupported_family_controls_are_not_accepted():
    with pytest.raises(TypeError):
        h3.TemporalGuide(("a.png",), 0, strength=1)
    for options in ({"seconds": 5}, {"fps": 30}, {"camera_lora": "static"},
                    {"last_frame": "a.png"}, {"audio_filename": "a.wav"}, {"steps": 4}):
        with pytest.raises(TypeError):
            h3.build_t2v("test", **options)


def test_h3_is_registered_on_the_video_backend(tmp_path):
    import dataclasses
    from fastapi.testclient import TestClient
    from easel.app import create_app
    from easel.config import Settings
    from tests.test_video_guidance import GuideComfy

    backend = GuideComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url="http://video",
                                   image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        model_ids = {entry["id"] for entry in client.get("/v1/models").json()["data"]}
        assert h3.VIDEO_MODEL_ID in model_ids
