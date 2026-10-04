"""Versioned discovery, fail-closed multipart admission and two-pass guide wiring."""
import dataclasses
import json
import io

from PIL import Image

import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from tests.test_video_loras import LoRAComfy


class GuideComfy(LoRAComfy):
    async def object_info(self, name):
        if name in self.missing_nodes:
            return {}
        keys = {
            'LTXVAddGuide': ['positive', 'negative', 'vae', 'latent', 'image', 'frame_idx', 'strength'],
            'LTXVCropGuides': ['positive', 'negative', 'latent'],
        }.get(name)
        if keys:
            self.info_calls += 1
            return {name: {'input': {'required': {key: [{'positive': 'CONDITIONING', 'negative': 'CONDITIONING', 'vae': 'VAE', 'latent': 'LATENT', 'image': 'IMAGE', 'frame_idx': 'INT', 'strength': 'FLOAT'}[key]] for key in keys}},
                           'output': ['CONDITIONING', 'CONDITIONING', 'LATENT']}}
        return await super().object_info(name)


@pytest.fixture
def api(tmp_path):
    backend = GuideComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url='http://video',
                                   image_job_dir=str(tmp_path), api_key='secret')
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        client.headers['Authorization'] = 'Bearer secret'
        yield client, backend


def still_png(color='red'):
    stream = io.BytesIO()
    Image.new('RGB', (8, 8), color).save(stream, format='PNG')
    return stream.getvalue()


def submit(client, guides=None, count=2, **fields):
    data = {'model': 'ltx-2.5', 'prompt': 'one coherent shot', 'seconds': '2',
            'size': '512x320', 'seed': '0', **fields}
    if guides is None:
        guides = [{'image_index': 0, 'frame_index': 0, 'strength': 0.7},
                  {'image_index': 1, 'frame_index': 47, 'strength': 0.3}]
    data['guiding_frames'] = guides if isinstance(guides, str) else json.dumps(guides)
    files = [('guiding_images', (f'{i}.png', still_png('red' if i % 2 else 'blue'), 'image/png'))
             for i in range(count)]
    return client.post('/v1/videos', data=data, files=files)


def nodes(graph, kind):
    return [(key, node['inputs']) for key, node in graph.items() if node['class_type'] == kind]


def test_capabilities_are_versioned_authenticated_and_read_only(api):
    client, backend = api
    response = client.get('/v1/videos/capabilities')
    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'video.capabilities'
    assert body['schema_version'] == 1
    assert body['model'] == 'ltx-2.5'
    assert body['seed']['max'] == str(2 ** 64 - 2)
    assert body['guiding_frames']['available'] is True
    assert body['guiding_frames']['validation'] == 'graph_contract_tested'
    assert body['guiding_frames']['frame_index_multiple'] == 1
    assert body['uploads']['max_total_bytes'] == 32 * 1024 * 1024
    assert backend.queue_calls == 0 and not backend.uploaded and backend.submitted_graph is None
    client.headers.pop('Authorization')
    assert client.get('/v1/videos/capabilities').status_code == 401


def test_guides_chain_video_only_and_crop_before_upscale_and_decode(api):
    client, backend = api
    assert submit(client, camera_lora='static').status_code == 200
    graph = backend.submitted_graph
    guides, crops = nodes(graph, 'LTXVAddGuide'), nodes(graph, 'LTXVCropGuides')
    assert len(guides) == 4 and len(crops) == 2
    assert [item['frame_idx'] for _, item in guides] == [0, 47, 0, 47]
    assert [item['strength'] for _, item in guides] == [0.7, 0.3, 0.7, 0.3]
    for offset in [0, 2]:
        first, second = guides[offset:offset + 2]
        assert second[1]['positive'] == [first[0], 0]
        assert second[1]['negative'] == [first[0], 1]
        assert second[1]['latent'] == [first[0], 2]
    assert guides[0][1]['latent'] == [nodes(graph, 'EmptyLTXVLatentVideo')[0][0], 0]
    assert guides[2][1]['positive'] == [crops[0][0], 0]
    assert guides[2][1]['negative'] == [crops[0][0], 1]
    assert guides[2][1]['latent'] == [nodes(graph, 'LTXVLatentUpsampler')[0][0], 0]
    assert nodes(graph, 'LTXVLatentUpsampler')[0][1]['samples'] == [crops[0][0], 2]
    assert nodes(graph, 'VAEDecodeTiled')[0][1]['samples'] == [crops[1][0], 2]
    for n, (_, concat) in enumerate(nodes(graph, 'LTXVConcatAVLatent')):
        assert concat['video_latent'] == [guides[n * 2 + 1][0], 2]
        assert crops[n][1]['positive'] == [guides[n * 2 + 1][0], 0]
    assert [item['noise_seed'] for _, item in nodes(graph, 'RandomNoise')] == [0, 1]
    assert len(backend.uploaded) == 2


def test_guides_sort_positions_without_changing_upload_mapping(api):
    client, backend = api
    response = submit(client, [{'image_index': 1, 'frame_index': 48}, {'image_index': 0, 'frame_index': 1}])
    assert response.status_code == 200
    guides = nodes(backend.submitted_graph, 'LTXVAddGuide')
    assert [item['frame_idx'] for _, item in guides] == [1, 48, 1, 48]
    assert all(item['strength'] == 1 for _, item in guides)


@pytest.mark.parametrize('guides,count', [
    ('garbage', 2), ('{}', 2), ([], 0),
    ([{'image_index': 0, 'frame_index': 0}], 2),
    ([{'image_index': 0, 'frame_index': 0}, {'image_index': 0, 'frame_index': 1}], 2),
    ([{'image_index': 0, 'frame_index': 0}, {'image_index': 1, 'frame_index': 0}], 2),
    ([{'image_index': 1, 'frame_index': 0}], 1),
    ([{'image_index': 0, 'frame_index': -1}], 1),
    ([{'image_index': 0, 'frame_index': 49}], 1),
    ([{'image_index': 0, 'frame_index': 0.5}], 1),
    ([{'image_index': True, 'frame_index': 0}], 1),
    ([{'image_index': 0, 'frame_index': True}], 1),
    ([{'image_index': 0, 'frame_index': 0, 'strength': True}], 1),
    ([{'image_index': 0, 'frame_index': 0, 'strength': float('nan')}], 1),
    ([{'image_index': 0, 'frame_index': 0, 'strength': 1.1}], 1),
    ([{'image_index': 0, 'frame_index': 0, 'path': '/etc/passwd'}], 1),
    ([{'image_index': i, 'frame_index': i} for i in range(9)], 9),
    ('[' * 1100 + ']' * 1100, 1),
])
def test_bad_guides_do_no_upstream_work(api, guides, count):
    client, backend = api
    response = submit(client, guides, count)
    assert response.status_code == 400
    assert backend.info_calls == backend.queue_calls == 0
    assert not backend.uploaded and backend.submitted_graph is None


@pytest.mark.parametrize('field,value', [('guide_strength', '1'), ('future_option', 'x')])
def test_unknown_video_fields_fail_instead_of_being_dropped(api, field, value):
    client, backend = api
    response = client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test', field: value})
    assert response.status_code == 400
    assert response.json()['error']['param'] == field
    assert backend.queue_calls == 0 and backend.submitted_graph is None


def test_duplicate_singleton_and_orphan_upload_fail(api):
    client, backend = api
    fields = [('model', (None, 'ltx-2.5')), ('prompt', (None, 'test')), ('prompt', (None, 'changed'))]
    assert client.post('/v1/videos', files=fields).status_code == 400
    assert client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test'},
                       files={'guiding_images': ('x.png', b'PNG', 'image/png')}).status_code == 400
    assert backend.info_calls == backend.queue_calls == 0


def test_guides_reject_first_image_or_ingredients_mode(api):
    client, backend = api
    for field in ['input_reference', 'lora_reference']:
        response = client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test',
            'guiding_frames': '[{"image_index":0,"frame_index":0}]'}, files=[
            ('guiding_images', ('guide.png', b'PNG', 'image/png')),
            (field, ('ref.png', b'PNG', 'image/png'))])
        assert response.status_code == 400
    assert submit(client, loras='[{"id":"ingredients"}]').status_code == 400
    assert backend.info_calls == backend.queue_calls == 0


def test_missing_guide_nodes_are_reported_and_fail_before_upload(api):
    client, backend = api
    backend.missing_nodes.add('LTXVAddGuide')
    cap = client.get('/v1/videos/capabilities').json()
    assert cap['guiding_frames']['available'] is False
    response = submit(client)
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'guiding_nodes_unavailable'
    assert backend.queue_calls == 0 and not backend.uploaded


def test_upload_type_and_total_size_fail_before_backend(api, monkeypatch):
    from easel import video_controls
    client, backend = api
    monkeypatch.setattr(video_controls, 'MAX_VIDEO_UPLOAD_BYTES', 5)
    assert submit(client).status_code == 400  # two 4-byte images exceed combined five
    assert backend.info_calls == backend.queue_calls == 0 and not backend.uploaded
    response = client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test',
        'guiding_frames': '[{"image_index":0,"frame_index":0}]'},
        files={'guiding_images': ('bad.mp4', b'bad', 'video/mp4')})
    assert response.status_code == 400


@pytest.mark.parametrize('kind', ['malformed', 'animated-png', 'animated-webp'])
def test_guides_require_decodable_single_frame_images(api, kind):
    client, backend = api
    if kind == 'malformed':
        payload, mime = b'PNG-not-an-image', 'image/png'
    else:
        stream = io.BytesIO()
        fmt = 'PNG' if kind == 'animated-png' else 'WEBP'
        Image.new('RGB', (8, 8), 'red').save(stream, format=fmt, save_all=True,
            append_images=[Image.new('RGB', (8, 8), 'blue')], duration=100, loop=0)
        payload, mime = stream.getvalue(), 'image/png' if fmt == 'PNG' else 'image/webp'
    response = client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test',
        'guiding_frames': '[{"image_index":0,"frame_index":0}]'},
        files={'guiding_images': ('guide.png', payload, mime)})
    assert response.status_code == 400
    assert response.json()['error']['param'] == 'guiding_images'
    expected = {
        'malformed': 'invalid guiding image: image could not be decoded safely',
        'animated-png': 'invalid guiding image: bytes do not match the declared image type',
        'animated-webp': 'invalid guiding image: guides must contain exactly one still image',
    }[kind]
    assert response.json()['error']['message'] == expected
    assert backend.info_calls == backend.queue_calls == 0 and not backend.uploaded


def test_truncated_jpeg_is_rejected_before_submission(api):
    client, backend = api
    stream = io.BytesIO()
    Image.new('RGB', (100, 100), 'red').save(stream, format='JPEG')
    payload = stream.getvalue()[:-2]
    response = client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test',
        'guiding_frames': '[{"image_index":0,"frame_index":0}]'},
        files={'guiding_images': ('guide.jpg', payload, 'image/jpeg')})
    assert response.status_code == 400
    assert response.json()['error']['message'] == (
        'invalid guiding image: image could not be decoded safely')
    assert response.json()['error']['param'] == 'guiding_images'
    assert backend.info_calls == backend.queue_calls == 0 and not backend.uploaded


@pytest.mark.parametrize('field', ['guiding_frames', 'loras', 'seed', 'seconds', 'size',
    'camera_lora', 'camera_lora_strength', 'motion_speed', 'lora_reference_strength'])
def test_explicit_empty_controls_are_not_silently_defaulted(api, field):
    client, backend = api
    response = client.post('/v1/videos', data={'model': 'ltx-2.5', 'prompt': 'test', field: ''})
    assert response.status_code == 400
    assert response.json()['error']['param'] == field
    assert backend.info_calls == backend.queue_calls == 0 and not backend.uploaded


@pytest.mark.parametrize('mutation', ['wrong-type', 'extra-required', 'narrow-frame-range',
    'narrow-strength-range', 'malformed-output', 'malformed-input', 'missing-input', 'bad-range', 'null-range'])
def test_incompatible_runtime_node_schemas_are_not_available(api, mutation):
    client, backend = api
    original = backend.object_info
    async def incompatible(name):
        result = await original(name)
        if name != 'LTXVAddGuide':
            return result
        node = result[name]
        required = node['input']['required']
        if mutation == 'wrong-type': required['image'] = ['STRING']
        if mutation == 'extra-required': required['unprovided'] = ['IMAGE']
        if mutation == 'narrow-frame-range': required['frame_idx'] = ['INT', {'min': 0, 'max': 10}]
        if mutation == 'narrow-strength-range': required['strength'] = ['FLOAT', {'min': 0.1, 'max': 1}]
        if mutation == 'malformed-output': node['output'] = None
        if mutation == 'malformed-input': node['input']['required'] = None
        if mutation == 'missing-input': del required['latent']
        if mutation == 'bad-range': required['strength'] = ['FLOAT', {'max': 'ten'}]
        if mutation == 'null-range': required['strength'] = ['FLOAT', {'min': None, 'max': 10}]
        return result
    backend.object_info = incompatible
    response = client.get('/v1/videos/capabilities')
    assert response.status_code == 200
    assert response.json()['guiding_frames']['available'] is False
    assert submit(client).status_code == 503
    assert backend.queue_calls == 0 and not backend.uploaded


def test_zero_guide_strength_is_preserved(api):
    client, backend = api
    assert submit(client, [{'image_index': 0, 'frame_index': 0, 'strength': 0}], count=1).status_code == 200
    assert [item['strength'] for _, item in nodes(backend.submitted_graph, 'LTXVAddGuide')] == [0, 0]
