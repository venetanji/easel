"""Qwen reference graphs must preserve each uploaded image's alpha channel."""
import pytest

from easel.qwen_graph import reference_edit


@pytest.mark.parametrize("reference_count", [1, 2, 16])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_reference_edit_rejoins_each_loaders_rgb_and_alpha(reference_count, batch_size):
    filenames = [f"reference_{index}.png" for index in range(reference_count)]
    graph = reference_edit(
        unet_name="qwen.safetensors", clip_name="clip.safetensors", vae_name="vae.safetensors",
        image_filenames=filenames, prompt="Keep the background transparent.",
        width=512, height=512, steps=25, batch_size=batch_size, seed=42,
    )
    encoder_id, encoder = next(
        (node_id, node) for node_id, node in graph.items()
        if node["class_type"] == "TextEncodeQwenImage21"
    )
    assert len([node for node in graph.values() if node["class_type"] == "JoinImageWithAlpha"]) == reference_count
    for index, filename in enumerate(filenames, start=1):
        rgba_id, output_slot = encoder["inputs"][f"images.image_{index}"]
        assert output_slot == 0
        joined = graph[rgba_id]
        assert joined["class_type"] == "JoinImageWithAlpha"
        image_id, image_slot = joined["inputs"]["image"]
        assert image_slot == 0
        assert graph[image_id] == {"class_type": "LoadImage", "inputs": {"image": filename}}
        # JoinImageWithAlpha already converts ComfyUI's inverse-alpha mask.
        assert joined["inputs"]["alpha"] == [image_id, 1]

    sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")
    latent_link = sampler["inputs"]["latent_image"]
    if batch_size > 1:
        repeated = graph[latent_link[0]]
        assert repeated == {"class_type": "RepeatLatentBatch",
                            "inputs": {"samples": [encoder_id, 2], "amount": batch_size}}
        assert latent_link[1] == 0
    else:
        assert latent_link == [encoder_id, 2]
