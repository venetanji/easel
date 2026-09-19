"""Unit tests for the vendored Flux.2 graph builders (no network)."""
import pytest

from easel.flux_graph import WorkflowGraph, text_to_image, reference_edit


# ---- helpers ----

def _all(graph, class_type):
    return [n for n in graph.values() if n["class_type"] == class_type]


def _one(graph, class_type):
    matches = _all(graph, class_type)
    assert len(matches) == 1, f"expected exactly one {class_type}, got {len(matches)}"
    return matches[0]


def _key(graph, class_type):
    for k, n in graph.items():
        if n["class_type"] == class_type:
            return k
    raise AssertionError(f"no {class_type} in graph")


def _iter_links(graph):
    for node in graph.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int):
                yield v


def assert_valid_graph(graph):
    keys = set(graph)
    assert keys, "graph is empty"
    for node in graph.values():
        assert set(node) == {"class_type", "inputs"}
        assert isinstance(node["class_type"], str)
        assert isinstance(node["inputs"], dict)
    for link in _iter_links(graph):
        assert link[0] in keys, f"dangling link {link}"


# ---- WorkflowGraph ----

def test_node_link_shape():
    g = WorkflowGraph()
    a = g.node("A", val=5)
    b = g.node("B", src=a[0])
    d = g.to_dict()
    a_key = a[0][0]
    assert d[a_key] == {"class_type": "A", "inputs": {"val": 5}}
    assert d[b[0][0]]["inputs"]["src"] == [a_key, 0]
    assert d[b[0][0]]["inputs"]["src"][0] in d


def test_output_slot_index_preserved():
    g = WorkflowGraph()
    a = g.node("A")
    assert a[0] == [a.id, 0]
    assert a[1] == [a.id, 1]


def test_to_dict_is_a_copy():
    g = WorkflowGraph()
    a = g.node("A", val=1)
    d = g.to_dict()
    d[a.id]["inputs"]["val"] = 999
    assert g.to_dict()[a.id]["inputs"]["val"] == 1


# ---- text_to_image ----

def test_text_to_image_selects_unet():
    g = text_to_image(unet_name="flux-2-klein-4b-fp8.safetensors", prompt="a cat", width=512, height=512)
    assert _one(g, "UNETLoader")["inputs"]["unet_name"] == "flux-2-klein-4b-fp8.safetensors"


def test_text_to_image_positive_and_empty_negative():
    g = text_to_image(unet_name="X", prompt="a red fox", width=512, height=512)
    texts = sorted(n["inputs"]["text"] for n in _all(g, "CLIPTextEncode"))
    assert texts == ["", "a red fox"]


def test_text_to_image_batch_size():
    g = text_to_image(unet_name="X", prompt="p", width=512, height=512, batch_size=3)
    assert _one(g, "EmptyFlux2LatentImage")["inputs"]["batch_size"] == 3


def test_text_to_image_dimensions_wired_to_latent_and_scheduler():
    g = text_to_image(unet_name="X", prompt="p", width=768, height=512)
    assert (_one(g, "EmptyFlux2LatentImage")["inputs"]["width"],
            _one(g, "EmptyFlux2LatentImage")["inputs"]["height"]) == (768, 512)
    assert (_one(g, "Flux2Scheduler")["inputs"]["width"],
            _one(g, "Flux2Scheduler")["inputs"]["height"]) == (768, 512)


def test_text_to_image_seed_wired():
    g = text_to_image(unet_name="X", prompt="p", width=512, height=512, seed=42)
    assert _one(g, "RandomNoise")["inputs"]["noise_seed"] == 42


def test_text_to_image_cfg_is_one():
    g = text_to_image(unet_name="X", prompt="p", width=512, height=512)
    assert _one(g, "CFGGuider")["inputs"]["cfg"] == 1.0


def test_text_to_image_clip_is_flux2():
    g = text_to_image(unet_name="X", prompt="p", width=512, height=512)
    assert _one(g, "CLIPLoader")["inputs"]["type"] == "flux2"


def test_text_to_image_uses_given_clip():
    g = text_to_image(unet_name="X", clip_name="my_clip.safetensors", prompt="p", width=512, height=512)
    assert _one(g, "CLIPLoader")["inputs"]["clip_name"] == "my_clip.safetensors"


def test_text_to_image_single_prompt_has_no_batcher():
    g = text_to_image(unet_name="X", prompt="just one", width=512, height=512)
    assert not _all(g, "SimplePromptBatcher")


def test_text_to_image_list_prompt_uses_batcher():
    g = text_to_image(unet_name="X", prompt=["a red fox", "a blue fox", "a green fox"],
                      width=512, height=512)
    batcher = _one(g, "SimplePromptBatcher")
    lines = [ln for ln in batcher["inputs"]["prompts"].split("\n") if ln]
    assert lines == ["a red fox", "a blue fox", "a green fox"]
    # positive encode is fed by the batcher; negative is a zero-out of it
    pos = _one(g, "CLIPTextEncode")
    assert pos["inputs"]["text"] == [_key(g, "SimplePromptBatcher"), 0]
    assert _all(g, "ConditioningZeroOut")


def test_text_to_image_singleton_list_is_single_prompt():
    g = text_to_image(unet_name="X", prompt=["only one"], width=512, height=512)
    assert not _all(g, "SimplePromptBatcher")
    assert "only one" in [n["inputs"]["text"] for n in _all(g, "CLIPTextEncode")]


def test_text_to_image_has_save_sink_and_valid_graph():
    g = text_to_image(unet_name="X", prompt="p", width=512, height=512)
    assert _all(g, "SaveImage")
    assert_valid_graph(g)


# ---- reference_edit ----

def test_reference_edit_negative_is_zeroout_single_encode():
    g = reference_edit(unet_name="X", image_filenames=["a.png"], prompt="p", width=512, height=512)
    assert _all(g, "ConditioningZeroOut")
    assert len(_all(g, "CLIPTextEncode")) == 1  # positive only


@pytest.mark.parametrize("k", [1, 2, 3])
def test_reference_edit_chains_reference_latent_per_ref_on_both_branches(k):
    g = reference_edit(unet_name="X", image_filenames=[f"{i}.png" for i in range(k)],
                       prompt="p", width=512, height=512)
    assert len(_all(g, "ReferenceLatent")) == 2 * k
    assert len(_all(g, "LoadImage")) == k
    assert len(_all(g, "VAEEncode")) == k


def test_reference_edit_loadimage_uses_given_filename():
    g = reference_edit(unet_name="X", image_filenames=["srv_123.png"], prompt="p", width=512, height=512)
    assert _one(g, "LoadImage")["inputs"]["image"] == "srv_123.png"


def test_reference_edit_uses_given_clip():
    g = reference_edit(unet_name="X", clip_name="my_clip.safetensors",
                       image_filenames=["a.png"], prompt="p", width=512, height=512)
    assert _one(g, "CLIPLoader")["inputs"]["clip_name"] == "my_clip.safetensors"


def test_reference_edit_list_prompt_uses_batcher():
    g = reference_edit(unet_name="X", image_filenames=["a.png"],
                       prompt=["turn it red", "make it night"], width=512, height=512)
    batcher = _one(g, "SimplePromptBatcher")
    lines = [ln for ln in batcher["inputs"]["prompts"].split("\n") if ln]
    assert lines == ["turn it red", "make it night"]
    assert _one(g, "CLIPTextEncode")["inputs"]["text"] == [_key(g, "SimplePromptBatcher"), 0]
    # still one ReferenceLatent per ref on each branch (refs encoded once)
    assert len(_all(g, "ReferenceLatent")) == 2


def test_reference_edit_single_prompt_has_no_batcher():
    g = reference_edit(unet_name="X", image_filenames=["a.png"], prompt="just one",
                       width=512, height=512)
    assert not _all(g, "SimplePromptBatcher")


def test_reference_edit_derives_size_from_first_ref_when_unset():
    g = reference_edit(unet_name="X", image_filenames=["a.png"], prompt="p")
    assert _all(g, "GetImageSize")
    w = _one(g, "EmptyFlux2LatentImage")["inputs"]["width"]
    assert isinstance(w, list) and w[0] == _key(g, "GetImageSize")


def test_reference_edit_explicit_size_skips_getimagesize():
    g = reference_edit(unet_name="X", image_filenames=["a.png"], prompt="p", width=512, height=768)
    assert not _all(g, "GetImageSize")
    latent = _one(g, "EmptyFlux2LatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (512, 768)


def test_reference_edit_batch_size():
    g = reference_edit(unet_name="X", image_filenames=["a.png"], prompt="p",
                       width=512, height=512, batch_size=4)
    assert _one(g, "EmptyFlux2LatentImage")["inputs"]["batch_size"] == 4


def test_reference_edit_empty_refs_raises():
    with pytest.raises(ValueError):
        reference_edit(unet_name="X", image_filenames=[], prompt="p", width=512, height=512)


def test_reference_edit_valid_graph():
    g = reference_edit(unet_name="X", image_filenames=["a.png", "b.png"], prompt="p", width=512, height=512)
    assert_valid_graph(g)
