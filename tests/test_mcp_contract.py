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

SOURCE = os.environ.get('EASEL_MEDIA_MCP_SOURCE')
if not SOURCE:
    pytest.skip('Set EASEL_MEDIA_MCP_SOURCE to the built easel-client checkout', allow_module_level=True)

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
