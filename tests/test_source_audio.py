"""Audio admission, source conditioning and fail-closed video HTTP contracts."""
import copy
import dataclasses
import io
import json
import subprocess
import wave

import pytest
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from easel import source_audio
from tests.test_h3_api import H3Comfy, png


AUDIO_SCHEMAS = {
    'LoadAudio': {'input': {'required': {'audio': [[], {'audio_upload': True}]}}, 'output': ['AUDIO']},
    'TrimAudioDuration': {'input': {'required': {'audio': ['AUDIO'], 'start_index': ['FLOAT'],
                                              'duration': ['FLOAT']}}, 'output': ['AUDIO']},
    'LTXVAudioVAEEncode': {'input': {'required': {'audio': ['AUDIO'], 'audio_vae': ['VAE']}}, 'output': ['LATENT']},
    'SolidMask': {'input': {'required': {'value': ['FLOAT'], 'width': ['INT'], 'height': ['INT']}}, 'output': ['MASK']},
    'SetLatentNoiseMask': {'input': {'required': {'samples': ['LATENT'], 'mask': ['MASK']}}, 'output': ['LATENT']},
}


def wav_bytes(seconds=6, *, sample_rate=24_000):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b'\0\0' * round(seconds * sample_rate))
    return output.getvalue()


@pytest.fixture
def audio_api(tmp_path):
    backend = H3Comfy()
    backend.schemas.update(copy.deepcopy(AUDIO_SCHEMAS))
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url='http://video',
                                   api_key='secret', image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        client.headers['Authorization'] = 'Bearer secret'
        yield client, backend


def submit(client, model, audio=None, *, mime='audio/wav', image=False, **extra):
    fields = {'model': model, 'prompt': 'a person singing', 'size': '512x320', **extra}
    if model == 'ltx-2.5':
        fields['seconds'] = '1'
    files = [('source_audio', ('private-original.wav', wav_bytes() if audio is None else audio, mime))]
    if image:
        files.append(('input_reference', ('anchor.png', png(), 'image/png')))
    return client.post('/v1/videos', data=fields, files=files)


def nodes(backend, kind):
    return [(name, node['inputs']) for name, node in backend.submitted_graph.items()
            if node['class_type'] == kind]


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('model,frames', [('ltx-2.5', 25), ('minimax-h3', 124)])
@pytest.mark.parametrize('image', [False, True])
def test_audio_conditions_video_and_muxes_normalized_source(audio_api, model, frames, image):
    client, backend = audio_api
    response = submit(client, model, image=image)
    assert response.status_code == 200, response.text
    assert response.json()['frames'] == frames
    name, trim = nodes(backend, 'TrimAudioDuration')[0]
    assert trim['duration'] == frames / 24 and trim['start_index'] == 0
    assert nodes(backend, 'CreateVideo')[0][1]['audio'] == [name, 0]
    assert not nodes(backend, 'VAEDecodeAudio') and not nodes(backend, 'LTXVAudioVAEDecode')
    audio_upload = next(item for item in backend.uploaded if item[2] == 'audio/wav')
    with wave.open(io.BytesIO(audio_upload[0]), 'rb') as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (48_000, 2, 2)
        assert audio.getnframes() == frames * 2000
    assert nodes(backend, 'LoadAudio')[0][1]['audio'] == audio_upload[1]
    assert audio_upload[1].startswith('easel_audio_') and 'private' not in audio_upload[1]
    if model == 'ltx-2.5':
        masks = nodes(backend, 'SetLatentNoiseMask')
        assert len(masks) == 2 and masks[0][1] == masks[1][1]
        assert nodes(backend, 'SolidMask')[0][1]['value'] == 0
        assert [node['audio_latent'] for _, node in nodes(backend, 'LTXVConcatAVLatent')] == [[key, 0] for key, _ in masks]
    else:
        guide_name, guide = nodes(backend, 'MiniMaxH3AddGuide')[0]
        assert guide['audio'] == [name, 0] and guide['frame_idx'] == 0
        assert nodes(backend, 'BasicGuider')[0][1]['conditioning'] == [guide_name, 0]


@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
@pytest.mark.parametrize('data,mime', [(b'', 'audio/wav'), (b'not audio', 'audio/wav'),
                                      (b'not audio', 'text/plain'), (wav_bytes(1), 'audio/flac')],
                         ids=['empty', 'corrupt', 'bad-mime', 'mislabeled'])
def test_invalid_audio_is_rejected_before_queue_or_upload(audio_api, model, data, mime):
    client, backend = audio_api
    response = submit(client, model, data, mime=mime)
    assert response.status_code == 400, response.text
    assert response.json()['error']['param'] == 'source_audio'
    assert not backend.uploaded and backend.queue_calls == 0 and backend.submitted_graph is None


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('model,seconds', [('ltx-2.5', 1), ('minimax-h3', 5)])
def test_audio_must_cover_rendered_duration_not_requested_seconds(audio_api, model, seconds):
    client, backend = audio_api
    response = submit(client, model, wav_bytes(seconds))
    assert response.status_code == 400 and response.json()['error']['code'] == 'audio_too_short'
    assert not backend.uploaded and backend.queue_calls == 0


@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_size_limit_before_decoder_or_backend(audio_api, model, monkeypatch):
    client, backend = audio_api
    monkeypatch.setattr(source_audio, 'MAX_AUDIO_BYTES', 64)
    response = submit(client, model, b'x' * 65)
    assert response.status_code == 400 and not backend.info_calls and not backend.uploaded


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_missing_audio_runtime_fails_before_queue_or_upload(audio_api, model):
    client, backend = audio_api
    del backend.schemas['LoadAudio']
    assert submit(client, model).status_code == 503
    assert not backend.uploaded and backend.queue_calls == 0 and backend.submitted_graph is None


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
@pytest.mark.parametrize('name,folder', [('../escape.wav', ''), ('x.wav [output]', ''), ('safe.wav', 'private')])
def test_unsafe_backend_audio_names_never_reach_submission(audio_api, model, name, folder):
    client, backend = audio_api

    async def unsafe_upload(*arguments):
        return name, folder

    backend.upload_image = unsafe_upload
    response = submit(client, model)
    assert response.status_code == 502 and response.json()['error']['code'] == 'upstream_upload_error'
    assert backend.submitted_graph is None


@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_audio_discovery_is_authenticated_and_read_only(audio_api, model):
    client, backend = audio_api
    response = client.get('/v1/videos/capabilities', params={'model': model})
    assert response.status_code == 200
    capability = response.json()['source_audio']
    assert capability['supported'] and capability['conditioning']
    assert capability['available'] == source_audio.audio_tools_available()
    assert capability['output'] == 'trimmed_source_audio'
    assert not backend.uploaded and backend.queue_calls == 0
    client.headers.pop('Authorization')
    assert submit(client, model, b'not audio').status_code == 401


def test_openapi_exposes_source_audio_upload(audio_api):
    client, _ = audio_api
    schema = client.get('/openapi.json').json()['components']['schemas']['VideoGenerationRequest']
    assert 'source_audio' in schema['properties']


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('extension,mime', [('mp3', 'audio/mpeg'), ('flac', 'audio/flac'),
                                         ('ogg', 'audio/ogg'), ('m4a', 'audio/mp4')])
def test_compressed_audio_formats_are_decoded(audio_api, tmp_path, extension, mime):
    client, backend = audio_api
    original = tmp_path / 'original.wav'
    original.write_bytes(wav_bytes())
    encoded = tmp_path / ('track.' + extension)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(original), str(encoded)],
                   check=True, timeout=15, capture_output=True)
    response = submit(client, 'ltx-2.5', encoded.read_bytes(), mime=mime)
    assert response.status_code == 200, response.text
    assert backend.uploaded[0][2] == 'audio/wav'


@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_repeated_audio_field_is_rejected(audio_api, model):
    client, backend = audio_api
    fields = {'model': model, 'prompt': 'singing'}
    response = client.post('/v1/videos', data=fields, files=[
        ('source_audio', ('a.wav', wav_bytes(), 'audio/wav')),
        ('source_audio', ('b.wav', wav_bytes(), 'audio/wav')),
    ])
    assert response.status_code == 400 and not backend.info_calls and not backend.uploaded


@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_missing_decoder_fails_before_backend(audio_api, model, monkeypatch):
    client, backend = audio_api

    async def missing_decoder(*arguments, **options):
        raise FileNotFoundError('test ffprobe')

    monkeypatch.setattr(source_audio.asyncio, 'create_subprocess_exec', missing_decoder)
    response = submit(client, model)
    assert response.status_code == 503 and response.json()['error']['code'] == 'audio_decoder_unavailable'
    assert not backend.info_calls and not backend.uploaded and backend.queue_calls == 0


@pytest.mark.parametrize('mutation', ['missing-input', 'list-output', 'extra-required', 'bad-bounds'])
def test_incompatible_audio_contract_is_unavailable(audio_api, mutation):
    client, backend = audio_api
    if mutation == 'missing-input':
        backend.schemas['LoadAudio']['input'] = None
    elif mutation == 'list-output':
        backend.schemas['LoadAudio']['output_is_list'] = [True]
    elif mutation == 'extra-required':
        backend.schemas['LoadAudio']['input']['required']['new_input'] = ['STRING']
    else:
        backend.schemas['TrimAudioDuration']['input']['required']['duration'] = ['FLOAT', {'max': 'bad'}]
    response = client.get('/v1/videos/capabilities')
    assert response.status_code == 200 and not response.json()['source_audio']['available']
    assert not backend.uploaded and backend.queue_calls == 0


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('kind', ['semantic', 'temporal'])
def test_h3_audio_preserves_reference_and_temporal_conditioning(audio_api, kind):
    client, backend = audio_api
    metadata = ({'semantic_references': json.dumps([{'image_index': 0}])} if kind == 'semantic' else
                {'temporal_groups': json.dumps([{'image_indices': [0], 'frame_index': 0}])})
    response = client.post('/v1/videos', data={'model': 'minimax-h3', 'prompt': 'Use <Picture 1>', **metadata},
                           files=[('images', ('anchor.png', png(), 'image/png')),
                                  ('source_audio', ('track.wav', wav_bytes(), 'audio/wav'))])
    assert response.status_code == 200, response.text
    assert nodes(backend, 'BasicScheduler')[0][1]['steps'] == 20
    guides = nodes(backend, 'MiniMaxH3AddGuide')
    assert len(guides) == (2 if kind == 'temporal' else 1)
    assert 'audio' in guides[-1][1]
    assert nodes(backend, 'BasicGuider')[0][1]['conditioning'] == [guides[-1][0], 0]


@pytest.mark.skipif(not source_audio.audio_tools_available(), reason='ffmpeg/ffprobe required')
@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_queue_rejection_never_uploads_audio(audio_api, model):
    client, backend = audio_api

    async def busy_queue():
        return {'queue_running': [], 'queue_pending': [[index, 'pending', {}] for index in range(100)]}

    backend.queue = busy_queue
    response = submit(client, model)
    assert response.status_code == 429, response.text
    assert not backend.uploaded and backend.submitted_graph is None
