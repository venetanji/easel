"""Curated guide lookup is authenticated, bounded, and independent of upstreams."""
import pytest

from tests.test_app import build


@pytest.mark.parametrize('model,mode,fragment', [
    ('qwen-image-2.1', 'generation', 'alpha channel'),
    ('qwen-image-2.1', 'editing', 'input order'),
    ('minimax-h3', 'base', 'integrated_multimodal_description:'),
    ('minimax-h3', 'reference', 'subject_definitions:'),
])
def test_reviewed_guide(model, mode, fragment):
    client, fake = build()
    response = client.get(f'/v1/models/{model}/guide', params={'mode': mode})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {'version', 'guide_id', 'model', 'mode', 'text', 'sources'}
    assert body['version'] == 1
    assert body['guide_id'] == f'{model}.{mode}'
    assert body['model'] == model and body['mode'] == mode
    assert fragment in body['text']
    assert body['sources'] and all(source.startswith('https://') for source in body['sources'])
    assert len(response.content) <= 16 * 1024
    assert fake.submitted_graph is None and fake.uploaded == [] and fake.fetched == []


def test_guide_requires_existing_auth():
    client, _ = build(api_key='secret')
    path = '/v1/models/qwen-image-2.1/guide?mode=generation'
    assert client.get(path).status_code == 401
    assert client.get(path, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert client.get(path, headers={'Authorization': 'Bearer secret'}).status_code == 200


@pytest.mark.parametrize('model,mode', [
    ('ltx-2.3', 'base'), ('minimax-h3', 'editing'), ('qwen-image-2.1', 'base'),
    ('qwen-image-2.1-pro', 'generation'), ('unknown', 'generation'),
    ('minimax-h3', 'temporal'), ('minimax-h3', ''),
])
def test_unreviewed_guide_is_404(model, mode):
    client, _ = build()
    response = client.get(f'/v1/models/{model}/guide', params={'mode': mode})
    assert response.status_code == 404


@pytest.mark.parametrize('mode', ['base', 'reference'])
def test_h3_dialogue_and_source_audio_boundary(mode):
    client, _ = build()
    response = client.get('/v1/models/minimax-h3/guide', params={'mode': mode})
    assert response.status_code == 200
    text = response.json()['text']
    assert '<d>[Language] ...</d>' in text
    assert 'outside' in text
    assert '48 kHz stereo' in text and 'trimmed' in text
    assert 'capabilities' in text and 'source_audio' in text
    assert 'Suno' in text


def test_registry_returns_same_payload_without_sharing_mutable_data():
    from easel.model_guides import get_model_guide
    client, _ = build()
    payload = get_model_guide('minimax-h3', 'base')
    assert client.get('/v1/models/minimax-h3/guide?mode=base').json() == payload
    payload['sources'].clear()
    assert get_model_guide('minimax-h3', 'base')['sources']


def test_missing_mode_is_explicit_and_auth_is_checked_first():
    client, _ = build(api_key='secret')
    path = '/v1/models/qwen-image-2.1/guide'
    assert client.get(path).status_code == 401
    response = client.get(path, headers={'Authorization': 'Bearer secret'})
    assert response.status_code == 400
    assert response.json()['error']['param'] == 'mode'


def test_openapi_requires_a_single_string_mode_parameter():
    client, _ = build()
    operation = client.get('/openapi.json').json()['paths']['/v1/models/{model_id}/guide']['get']
    modes = [parameter for parameter in operation['parameters']
             if parameter['name'] == 'mode' and parameter['in'] == 'query']
    assert len(modes) == 1
    assert modes[0]['required'] is True
    assert modes[0]['schema'] == {'type': 'string', 'title': 'Mode'}
