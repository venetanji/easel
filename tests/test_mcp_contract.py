"""Opt-in cross-language Media MCP serialization → actual Easel API → graph.

Build easel-client's media-mcp first and set EASEL_MEDIA_MCP_SOURCE to that repo.
Node captures the production FormData through an injected fetch; Python submits
those exact parts to the real FastAPI app. No sockets, credentials or GPU work.
"""
import base64
import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from easel.app import create_app
from easel.config import Settings
from easel.video_loras import VIDEO_LORAS
from tests.test_video_guidance import GuideComfy
from tests.test_audio_api import audio_client
from tests.test_source_audio import audio_api, wav_bytes, nodes
from easel import source_audio
import wave

SOURCE = os.environ.get('EASEL_MEDIA_MCP_SOURCE')
if not SOURCE:
    pytest.skip('Set EASEL_MEDIA_MCP_SOURCE to the built easel-client checkout', allow_module_level=True)

if not (Path(SOURCE) / 'packages/media-mcp/dist/video.js').is_file():
    raise RuntimeError('EASEL_MEDIA_MCP_SOURCE must be a built easel-client checkout')

NODE_SERIALIZE = r'''
const {pathToFileURL} = await import('node:url');
const {readFileSync} = await import('node:fs');
const {generateVideo} = await import(pathToFileURL(process.argv[1]).href);
const input = JSON.parse(readFileSync(0, 'utf8'));
let calls = 0, parts;
await generateVideo({...input, baseUrl:'https://easel.test', fetchImpl:async(url, init)=>{
  if (url !== 'https://easel.test/v1/videos' || init.method !== 'POST') throw new Error('unexpected request');
  calls++;
  parts = await Promise.all([...init.body.entries()].map(async([field,value]) =>
    typeof value === 'string' ? {field, value} :
      {field, name:value.name, type:value.type, data:Buffer.from(await value.arrayBuffer()).toString('base64')}));
  return Response.json({id:'video_contract',object:'video',status:'queued',model:'ltx-2.5'});
}});
if(calls !== 1) throw new Error('generation must submit exactly once');
process.stdout.write(JSON.stringify(parts));
'''
NODE_DISCOVER = r'''
const {pathToFileURL} = await import('node:url');
const {readFileSync} = await import('node:fs');
const provider = await import(pathToFileURL(process.argv[1]).href);
const {capabilities, catalog} = JSON.parse(readFileSync(0, 'utf8'));
const options = {model:'ltx-2.5',baseUrl:'https://easel.test',fetchImpl:async(url,init)=>{
  if(init.method !== 'GET') throw new Error('discovery must be read-only');
  return Response.json(url.endsWith('/capabilities') ? capabilities : catalog);
}};
const caps=await provider.discoverVideoCapabilities(options);
const loras=await provider.listVideoLoras(options);
process.stdout.write(JSON.stringify({caps,loras}));
'''


def node_run(script, data):
    entry = Path(SOURCE) / 'packages/media-mcp/dist/video.js'
    assert entry.is_file(), 'Build Media MCP before this opt-in contract suite'
    result = subprocess.run(['node', '--input-type=module', '-e', script, str(entry)],
                            input=json.dumps(data), text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture
def api(tmp_path):
    backend = GuideComfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url='http://video',
                                   image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        yield client, backend


def still():
    stream = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(stream, 'PNG')
    return {'data': base64.b64encode(stream.getvalue()).decode(), 'mimeType': 'image/png', 'name': 'guide.png'}


@pytest.mark.parametrize('mode', ['guides', 'ingredients', 'slow-motion'])
def test_actual_mcp_formdata_reaches_matching_graph(api, mode):
    client, backend = api
    image = still()
    options = {'prompt': 'test', 'model': 'ltx-2.5', 'size': '512x320', 'seconds': 2,
               'seed': str(2 ** 64 - 2)}
    if mode == 'guides':
        options.update(cameraLora='static', cameraLoraStrength=0.6,
            guidingFrames=[{'image': image, 'frameIndex': 1, 'strength': 0},
                           {'image': image, 'frameIndex': 48, 'strength': 0.5}])
    elif mode == 'ingredients':
        options.update(seconds=5, loras=[{'id': 'ingredients', 'strength': 0.8}],
                       loraReference=image, loraReferenceStrength=0.4)
    else:
        options.update(loras=[{'id': 'slow-motion', 'strength': 0.7}],
                       motionSpeed=0.2, inputReference=image)
    parts = node_run(NODE_SERIALIZE, options)
    files = [(part['field'], (part['name'], base64.b64decode(part['data']), part['type'])
              if 'data' in part else (None, part['value'])) for part in parts]
    response = client.post('/v1/videos', files=files)
    assert response.status_code == 200, response.text
    graph = backend.submitted_graph
    assert [node['inputs']['noise_seed'] for node in graph.values()
            if node['class_type'] == 'RandomNoise'] == [2 ** 64 - 2, 2 ** 64 - 1]
    if mode == 'guides':
        guides = [node['inputs'] for node in graph.values() if node['class_type'] == 'LTXVAddGuide']
        assert [(node['frame_idx'], node['strength']) for node in guides] == [(1, 0), (48, .5)] * 2
    elif mode == 'ingredients':
        guides = [node['inputs'] for node in graph.values() if node['class_type'] == 'LTXAddVideoICLoRAGuide']
        assert [node['strength'] for node in guides] == [.4, .4]
    else:
        assert next(node['inputs']['frame_rate'] for node in graph.values()
                    if node['class_type'] == 'LTXVConditioning') == 120


def test_actual_server_discovery_is_consumed_by_media_mcp(api):
    client, _ = api
    payload = node_run(NODE_DISCOVER, {'capabilities': client.get('/v1/videos/capabilities').json(),
                                      'catalog': client.get('/v1/videos/loras').json()})
    assert payload['caps']['guiding_frames']['available'] is True
    assert {entry['id'] for entry in payload['loras']} == set(VIDEO_LORAS)
    assert any(not entry['supported'] for entry in payload['loras'])


def test_h3_receipt_is_accepted_by_production_media_job_parser(tmp_path):
    """Timing extensions must remain usable by the published poll/content client."""
    from tests.test_h3_api import H3Comfy
    backend = H3Comfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url='http://video',
                                   image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        response = client.post('/v1/videos', data={'model':'minimax-h3','prompt':'shot','frames':'141'})
        assert response.status_code == 200
        receipt = response.json()
        script = r'''
const {pathToFileURL} = await import('node:url');
const {readFileSync} = await import('node:fs');
const {parseMediaJob} = await import(new URL('./media-job.js', pathToFileURL(process.argv[1])).href);
const input = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(parseMediaJob(input, undefined, input.id)));
'''
        assert node_run(script, receipt)['id'] == receipt['id']
        assert node_run(script, client.get('/v1/videos/'+receipt['id']).json())['status'] == 'queued'


@pytest.mark.parametrize('mode', ['text', 'image', 'semantic', 'temporal'])
def test_h3_mcp_submission_reaches_actual_native_graph(tmp_path, mode):
    from tests.test_h3_api import H3Comfy
    backend = H3Comfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url='http://video',
                                   image_job_dir=str(tmp_path))
    options = {'model': 'minimax-h3', 'prompt': 'shot', 'frames': 141, 'seed': str(2**64-1)}
    if mode == 'image':
        options['inputReference'] = still()
    elif mode in ('semantic', 'temporal'):
        options['semanticReferences'] = [still()]
        if mode == 'temporal':
            options['temporalGroups'] = [{'frameIndex': 17, 'images': [still()]}]
    parts = node_run(NODE_SERIALIZE, options)
    assert not any(part['field'] == 'seconds' for part in parts)
    files = [(part['field'], (part['name'], base64.b64decode(part['data']), part['type'])
              if 'data' in part else (None, part['value'])) for part in parts]
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        response = client.post('/v1/videos', files=files)
        assert response.status_code == 200, response.text
        assert response.json()['frames'] == 141
    assert next(node['inputs']['noise_seed'] for node in backend.submitted_graph.values()
                if node['class_type'] == 'RandomNoise') == 2**64-1
    assert len(backend.uploaded) == {'text': 0, 'image': 1, 'semantic': 1, 'temporal': 2}[mode]


NODE_AUDIO = r'''
const {pathToFileURL} = await import('node:url');
const {readFileSync} = await import('node:fs');
const {generateAudio} = await import(new URL('./audio.js', pathToFileURL(process.argv[1])).href);
const input = JSON.parse(readFileSync(0, 'utf8'));
let payload, calls = 0;
await generateAudio(input.kind, input.args, {baseUrl:'https://easel.test', fetchImpl:async(url,init)=>{
  if(url !== 'https://easel.test/v1/audio/generations' || init.method !== 'POST') throw new Error('unexpected request');
  calls++; payload = JSON.parse(init.body); return Response.json({status:'submitted'});
}});
if(calls !== 1) throw new Error('submit exactly once');
process.stdout.write(JSON.stringify(payload));
'''


@pytest.mark.parametrize('kind,args', [
    ('music', {'prompt': 'jazz', 'lyrics': 'hello', 'title': 'Test', 'sunoModel': 'v5',
               'weirdness': 20, 'styleInfluence': 60, 'durationSeconds': 120, 'dryRun': True}),
    ('music', {'prompt': 'jazz', 'referenceAudioId': '11111111-1111-4111-8111-111111111111',
               'inspirationIds': [], 'dryRun': True}),
    ('music', {'prompt': 'jazz', 'inspirationPlaylist': 'playlist',
               'inspirationIds': [], 'dryRun': True}),
    ('speech', {'prompt': 'Hello', 'tone': 'warm', 'backgroundMusic': True,
                'vocalGender': 'female', 'variety': 2, 'dryRun': True}),
    ('sound', {'prompt': 'rain', 'soundType': 'loop', 'bpm': 100, 'dryRun': True}),
])
async def test_audio_mcp_payload_reaches_actual_easel_api(audio_client, kind, args):
    client, backend = audio_client
    payload = node_run(NODE_AUDIO, {'kind': kind, 'args': args})
    response = await client.post('/v1/audio/generations', json=payload)
    assert response.status_code == 202, response.text
    assert response.json()['model'] == 'suno-' + kind
    assert backend.calls[-1].url.path == '/api/v1/generate/' + kind
    assert json.loads(backend.calls[-1].content)['dry_run'] is True


def test_mcp_audio_schema_covers_every_api_request_field():
    import re
    from easel.audio_api import MusicGeneration, SpeechGeneration, SoundGeneration
    script = r'''
const {pathToFileURL} = await import('node:url');
const schemas = await import(new URL('./audio.js', pathToFileURL(process.argv[1])).href);
process.stdout.write(JSON.stringify(Object.fromEntries(['Music','Speech','Sound'].map(kind=>
  [kind, Object.keys(schemas[kind+'InputSchema'].shape)]))));
'''
    fields = node_run(script, {})
    for kind, model in [('Music', MusicGeneration), ('Speech', SpeechGeneration), ('Sound', SoundGeneration)]:
        wire = {re.sub(r'([A-Z])', lambda match: '_' + match[1].lower(), field) for field in fields[kind]}
        assert wire == set(model.model_fields)


NODE_SOURCE_AUDIO = r"""
const {pathToFileURL} = await import('node:url');
const {readFileSync} = await import('node:fs');
const {generateVideo} = await import(pathToFileURL(process.argv[1]).href);
const {options, capabilities, loseReceipt=false} = JSON.parse(readFileSync(0,'utf8'));
let calls=0, reads=0, parts=[], error;
try {
 await generateVideo({...options,baseUrl:'https://easel.test',apiKey:'secret',fetchImpl:async(url,init)=>{
  if(init.headers.Authorization !== 'Bearer secret') throw new Error('missing authentication');
  if(init.method==='GET') {
   if(url!=='https://easel.test/v1/videos/capabilities'+(options.model==='minimax-h3'?'?model=minimax-h3':'')) throw new Error('wrong capability model');
   reads++; return Response.json(capabilities);
  }
  if(url!=='https://easel.test/v1/videos' || init.method!=='POST') throw new Error('unexpected request');
  calls++;
  parts=await Promise.all([...init.body.entries()].map(async([field,value])=>typeof value==='string'
    ? {field,value} : {field,name:value.name,type:value.type,data:Buffer.from(await value.arrayBuffer()).toString('base64')}));
  if(loseReceipt) throw new Error('receipt lost after acceptance');
  return Response.json({id:'video_contract',object:'video',status:'queued',model:options.model});
 }});
} catch(e) { error=e.message; }
process.stdout.write(JSON.stringify({calls,reads,parts,error}));
"""


def audio_options(model, seconds=6):
    options = {'model': model, 'prompt': 'visible articulation and closed lips in pauses',
               'size': '768x512' if model == 'minimax-h3' else '512x320', 'inputReference': still(),
               'sourceAudio': {'data': base64.b64encode(wav_bytes(seconds)).decode(),
                               'mimeType': 'audio/wav', 'name': 'speech.wav'}}
    options.update({'frames': 124} if model == 'minimax-h3' else {'seconds': 5})
    return options


def post_parts(client, parts):
    return client.post('/v1/videos', files=[
        (part['field'], (part['name'], base64.b64decode(part['data']), part['type'])
         if 'data' in part else (None, part['value'])) for part in parts])


@pytest.mark.parametrize('model,frames', [('ltx-2.5', 121), ('minimax-h3', 124)])
@pytest.mark.parametrize('lose_receipt', [False, True])
def test_mcp_source_audio_reaches_normalization_conditioning_and_source_mux(audio_api, model, frames, lose_receipt):
    assert source_audio.audio_tools_available(), 'ffmpeg and ffprobe required for configured contract'
    client, backend = audio_api
    capabilities = client.get('/v1/videos/capabilities', params={'model': model}).json()
    result = node_run(NODE_SOURCE_AUDIO, {'options': audio_options(model),
        'capabilities': capabilities, 'loseReceipt': lose_receipt})
    assert result['calls'] == result['reads'] == 1, result
    assert bool(result.get('error')) == lose_receipt
    parts = result['parts']
    assert sum(part['field'] == 'source_audio' for part in parts) == 1
    assert sum(part['field'] == 'input_reference' for part in parts) == 1
    if model == 'minimax-h3':
        assert not any(part['field'] == 'seconds' for part in parts)
        assert next(part['value'] for part in parts if part['field'] == 'frames') == '124'
    response = post_parts(client, parts)
    assert response.status_code == 200, response.text
    assert response.json()['frames'] == frames
    key, trim = nodes(backend, 'TrimAudioDuration')[0]
    assert trim['duration'] == frames / 24 and trim['start_index'] == 0
    assert nodes(backend, 'CreateVideo')[0][1]['audio'] == [key, 0]
    assert not nodes(backend, 'VAEDecodeAudio') and not nodes(backend, 'LTXVAudioVAEDecode')
    data, name, _ = next(upload for upload in backend.uploaded if upload[2] == 'audio/wav')
    with wave.open(io.BytesIO(data), 'rb') as decoded:
        assert (decoded.getframerate(), decoded.getnchannels(), decoded.getnframes()) == (48000, 2, frames * 2000)
    assert nodes(backend, 'LoadAudio')[0][1]['audio'] == name
    if model == 'minimax-h3':
        assert nodes(backend, 'MiniMaxH3AddGuide')[-1][1]['audio'] == [key, 0]
    else:
        assert len(nodes(backend, 'SetLatentNoiseMask')) == 2


@pytest.mark.parametrize('model', ['ltx-2.5', 'minimax-h3'])
def test_mcp_short_audio_rejected_before_backend(audio_api, model):
    client, backend = audio_api
    result = node_run(NODE_SOURCE_AUDIO, {'options': audio_options(model, 5),
        'capabilities': client.get('/v1/videos/capabilities', params={'model': model}).json()})
    assert result['calls'] == 1 and not result.get('error'), result
    response = post_parts(client, result['parts'])
    assert response.status_code == 400 and response.json()['error']['code'] == 'audio_too_short'
    assert not backend.uploaded and backend.queue_calls == 0


@pytest.mark.parametrize('old', [False, True])
def test_mcp_old_or_unavailable_capabilities_block_only_supplied_audio(audio_api, old):
    client, backend = audio_api
    del backend.schemas['LoadAudio']
    cap = client.get('/v1/videos/capabilities', params={'model': 'minimax-h3'}).json()
    if old:
        cap.pop('source_audio')
    options = audio_options('minimax-h3')
    result = node_run(NODE_SOURCE_AUDIO, {'options': options, 'capabilities': cap})
    assert result['calls'] == 0 and 'Source audio' in result['error']
    options.pop('sourceAudio')
    result = node_run(NODE_SOURCE_AUDIO, {'options': options, 'capabilities': cap})
    assert result['reads'] == 0 and result['calls'] == 1 and not result.get('error')
    assert post_parts(client, result['parts']).status_code == 200


def test_mcp_source_audio_definitions_match_server():
    script = r"""
const {pathToFileURL} = await import('node:url');
const audio = await import(new URL('./source-audio-upload.js',pathToFileURL(process.argv[1])).href);
const {VideoGenerationSchema} = await import(new URL('./video-input.js',pathToFileURL(process.argv[1])).href);
process.stdout.write(JSON.stringify({mime:audio.SOURCE_AUDIO_MIME_TYPES,max:audio.MAX_SOURCE_AUDIO_BYTES,
 fields:Object.keys(VideoGenerationSchema.shape)}));
"""
    definitions = node_run(script, {})
    cap = source_audio.source_audio_capabilities(True)
    assert definitions['mime'] == cap['mime_types'] == list(source_audio.AUDIO_MIME_TYPES)
    assert definitions['max'] == cap['max_bytes'] == source_audio.MAX_AUDIO_BYTES
    assert cap['field'] == 'source_audio' and 'sourceAudio' in definitions['fields']


@pytest.mark.parametrize('model,mode', [('qwen-image-2.1','generation'),
    ('qwen-image-2.1','editing'), ('minimax-h3','base'), ('minimax-h3','reference')])
def test_mcp_guide_preserves_actual_authenticated_server_identity(audio_api, model, mode):
    client, backend = audio_api
    response = client.get(f'/v1/models/{model}/guide', params={'mode': mode})
    assert response.status_code == 200
    script = r"""
const {pathToFileURL} = await import('node:url');
const {readFileSync} = await import('node:fs');
const {getModelGuide} = await import(new URL('./model-guides.js',pathToFileURL(process.argv[1])).href);
const {model,mode,payload}=JSON.parse(readFileSync(0,'utf8'));
const guide=await getModelGuide({model,mode,baseUrl:'https://easel.test',apiKey:'secret',fetchImpl:async(url,init)=>{
 if(url!==`https://easel.test/v1/models/${model}/guide?mode=${mode}` || init.method!=='GET'
   || init.headers.Authorization!=='Bearer secret' || init.redirect!=='error') throw new Error('wrong guide request');
 return Response.json(payload);
}});
process.stdout.write(JSON.stringify(guide));
"""
    assert node_run(script, {'model': model, 'mode': mode, 'payload': response.json()}) == response.json()
    assert not backend.uploaded and backend.submitted_graph is None
