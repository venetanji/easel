"""Offline multipart H3 integration against captured native Comfy contracts."""
import copy
import dataclasses
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings
from tests.test_video_duration import FakeVideoComfy
from tests.test_comfy_client import INVALID_UPLOAD_RESPONSES

SCHEMAS = json.loads((Path(__file__).parent / 'fixtures/h3/runtime_nodes.json').read_text())['nodes']

class H3Comfy(FakeVideoComfy):
    def __init__(self):
        super().__init__()
        self.schemas = copy.deepcopy(SCHEMAS)
        self.info_calls = []
        self.history = None

    async def object_info(self, name):
        self.info_calls.append(name)
        return {name: self.schemas[name]} if name in self.schemas else {}

    async def upload_image(self, data, filename, content_type):
        self.uploaded.append((data, filename, content_type))
        return filename, ''

    async def history_item(self, prompt_id):
        return self.history

    async def queue(self):
        self.queue_calls += 1
        return {'queue_pending': [[0, 'duration-test-prompt-id', self.submitted_graph]] if self.submitted_graph else [], 'queue_running': []}

    async def fetch(self, ref):
        return b'mp4-output'

@pytest.fixture
def api(tmp_path):
    backend = H3Comfy()
    settings = dataclasses.replace(Settings.from_env({}), comfy_video_url='http://video', api_key='secret', image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings, comfy=backend, comfy_video=backend)) as client:
        client.headers['Authorization'] = 'Bearer secret'
        yield client, backend

def png(size=(8, 8), fmt='PNG'):
    out = io.BytesIO()
    Image.new('RGB', size, 'red').save(out, format=fmt)
    return out.getvalue()

def submit(client, files=None, **fields):
    return client.post('/v1/videos', data={'model': 'minimax-h3', 'prompt': 'one shot', **fields}, files=files)

def nodes(backend, kind):
    return [n['inputs'] for n in backend.submitted_graph.values() if n['class_type'] == kind]

def test_registration_capabilities_and_model_selection(api):
    client, backend = api
    assert 'minimax-h3' in {m['id'] for m in client.get('/v1/models').json()['data']}
    response = client.get('/v1/videos/capabilities?model=minimax-h3')
    assert response.status_code == 200
    cap = response.json()
    assert cap['model'] == 'minimax-h3'
    assert cap['frames'] == {'min': 124, 'max': 362, 'default': 124, 'step': 17, 'offset': 5}
    assert cap['seed']['max'] == str(2**64-1)
    assert cap['profiles']['fl8']['available']
    assert cap['profiles']['temporal_ref20']['available']
    assert cap['profiles']['temporal_ref20']['validation'] == 'graph_contract_tested'
    assert client.get('/v1/videos/capabilities?model=unknown').status_code == 400
    assert not backend.uploaded and backend.queue_calls == 0 and backend.submitted_graph is None
    client.headers.pop('Authorization')
    assert client.get('/v1/videos/capabilities?model=minimax-h3').status_code == 401
    assert submit(client).status_code == 401

def test_t2v_exact_frames_seed_and_durable_receipt(api):
    client, backend = api
    response = submit(client, frames='141', seed=str(2**64-1))
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert (receipt['frames'], receipt['fps'], receipt['duration']) == (141, 24, 141/24)
    assert nodes(backend, 'RandomNoise')[0]['noise_seed'] == 2**64-1
    assert nodes(backend, 'MiniMaxH3ImageToVideo')[0]['length'] == 141
    assert nodes(backend, 'BasicScheduler')[0]['steps'] == 8
    assert client.get('/v1/videos/'+receipt['id']).json()['frames'] == 141
    backend.history = {'status': {'completed': True}, 'outputs': {'save': {'videos': [{'filename':'h3.mp4','type':'output'}]}}}
    assert client.get('/v1/videos/'+receipt['id']).json()['duration'] == 141/24
    assert client.get('/v1/videos/'+receipt['id']+'/content').content == b'mp4-output'
    assert backend.queue_calls == 2

@pytest.mark.parametrize('fields', [dict(seconds='5'),dict(frames='125'),dict(frames='123'),dict(frames='379'),dict(seed=str(2**64)),dict(size='1280x720'),dict(camera_lora='static'),dict(negative_prompt='x'),dict(frames='true'),dict(semantic_references='[]'),dict(temporal_groups='[]'),dict(input_audio='file.wav')])
def test_invalid_input_has_zero_upstream_work(api, fields):
    client, backend = api
    assert submit(client, **fields).status_code == 400
    assert not backend.info_calls and backend.queue_calls == 0 and not backend.uploaded and backend.submitted_graph is None

@pytest.mark.parametrize('mode', ['i2v','semantic','temporal','mixed'])
def test_upload_modes_and_profile_separation(api, mode):
    client, backend = api
    fields, files = {}, []
    if mode == 'i2v':
        files = [('input_reference', ('../../client.png', png(), 'image/png'))]
    else:
        count = 6 if mode == 'mixed' else 5 if mode == 'temporal' else 2
        files = [('images', (f'../client{i}.png', png(), 'image/png')) for i in range(count)]
        if mode in ('semantic','mixed'):
            fields['semantic_references'] = json.dumps([{'image_index':i} for i in range(1 if mode == 'mixed' else 2)])
        if mode in ('temporal','mixed'):
            fields['temporal_groups'] = json.dumps([{'frame_index':0,'image_indices':list(range(1 if mode == 'mixed' else 0,count))}])
    response = submit(client, files=files, **fields)
    assert response.status_code == 200, response.text
    assert len(backend.uploaded) == len(files)
    assert all('/' not in name and name.endswith('.png') for _,name,_ in backend.uploaded)
    if mode == 'i2v':
        assert nodes(backend, 'MiniMaxH3ImageToVideo')[0]['first_frame']
    else:
        assert nodes(backend, 'BasicScheduler')[0]['steps'] == 20
        assert not nodes(backend, 'MiniMaxH3SigmaShift') and not nodes(backend,'LoraLoaderModelOnly')
        assert len(nodes(backend,'MiniMaxH3AddGuide')) == int(mode in ('temporal','mixed'))

@pytest.mark.parametrize('metadata', [
 {'semantic_references':'[{"image_index":true}]'},
 {'semantic_references':'[{"image_index":0,"strength":1}]'},
 {'semantic_references':'[{"image_index":0},{"image_index":0}]'},
 {'temporal_groups':'[{"frame_index":0,"image_indices":[0,1]}]'},
 {'temporal_groups':'[{"frame_index":0,"image_indices":[0],"strength":1}]'},
 {'temporal_groups':'[{"frame_index":124,"image_indices":[0]}]'},
 {'temporal_groups':'[{"frame_index":0,"image_indices":[0]},{"frame_index":0,"image_indices":[1]}]'},
 {'semantic_references':'[{"image_index":0}]'},
])
def test_bad_metadata_or_orphan_rejected_before_discovery(api, metadata):
    client, backend = api
    files = [('images',(f'{i}.png',png(),'image/png')) for i in range(2)]
    assert submit(client, files=files, **metadata).status_code == 400
    assert not backend.info_calls and backend.queue_calls == 0 and not backend.uploaded

@pytest.mark.parametrize('fault', ['missing','asset','type','output','new_required','bound','enum','dynamic'])
def test_runtime_contract_failure_precedes_queue_and_upload(api, fault):
    client, backend = api
    if fault == 'missing': del backend.schemas['MiniMaxH3ImageToVideo']
    elif fault == 'asset': backend.schemas['UNETLoader']['input']['required']['unet_name'][0]=[]
    elif fault == 'type': backend.schemas['MiniMaxH3ImageToVideo']['input']['required']['clip'][0]='IMAGE'
    elif fault == 'output': backend.schemas['MiniMaxH3ImageToVideo']['output']=['LATENT','CONDITIONING']
    elif fault == 'new_required': backend.schemas['MiniMaxH3ImageToVideo']['input']['required']['new']=['INT']
    elif fault == 'bound': backend.schemas['RandomNoise']['input']['required']['noise_seed'][1]['max']=2**32
    elif fault == 'enum': backend.schemas['KSamplerSelect']['input']['required']['sampler_name'][1]['options']=[]
    else: backend.schemas['SaveVideo']['input']['required']['format'][1]['options']=[]
    assert submit(client, seed=str(2**64-1), files=[('input_reference',('a.png',png(),'image/png'))]).status_code == 503
    assert backend.queue_calls == 0 and not backend.uploaded and backend.submitted_graph is None

@pytest.mark.parametrize('data,mime', [(b'bad','image/png'),(png(fmt='JPEG'),'image/png'),(png(),'image/webp')])
def test_stills_decode_and_type_validation(api, data, mime):
    client, backend = api
    assert submit(client, files=[('input_reference',('a.png',data,mime))]).status_code == 400
    assert not backend.info_calls and not backend.uploaded and backend.queue_calls == 0

def test_temporal_dimensions_match(api):
    client, backend = api
    files=[('images',(f'{i}.png',png((8+i,8)),'image/png')) for i in range(5)]
    assert submit(client,files=files,temporal_groups='[{"frame_index":0,"image_indices":[0,1,2,3,4]}]').status_code == 400
    assert not backend.info_calls and backend.queue_calls == 0

def test_duplicate_form_fields_rejected(api):
    client, backend = api
    files = [('model',(None,'minimax-h3')),('prompt',(None,'shot')),('frames',(None,'124')),('frames',(None,'141'))]
    assert client.post('/v1/videos', files=files).status_code == 400
    assert not backend.info_calls and backend.queue_calls == 0

def test_animation_rejected(api):
    client, backend = api
    out=io.BytesIO()
    Image.new('RGB',(8,8),'red').save(out,format='PNG',save_all=True,append_images=[Image.new('RGB',(8,8),'blue')])
    assert submit(client,files=[('input_reference',('animation.png',out.getvalue(),'image/png'))]).status_code == 400
    assert not backend.info_calls and backend.queue_calls == 0

@pytest.mark.parametrize('limit', ['bytes','each_pixels','total_pixels'])
def test_upload_resource_limits(api, monkeypatch, limit):
    from easel import h3_controls
    client, backend = api
    count = 2 if limit == 'total_pixels' else 1
    monkeypatch.setattr(h3_controls, {'bytes':'MAX_BYTES','each_pixels':'MAX_IMAGE_PIXELS','total_pixels':'MAX_PIXELS'}[limit], {'bytes':len(png())-1,'each_pixels':63,'total_pixels':127}[limit])
    files=[('images',(f'{i}.png',png(),'image/png')) for i in range(count)]
    assert submit(client,files=files,semantic_references=json.dumps([{'image_index':i} for i in range(count)])).status_code == 400
    assert not backend.info_calls and backend.queue_calls == 0 and not backend.uploaded

def test_queue_full_never_uploads(api):
    client, backend = api
    async def full_queue():
        backend.queue_calls += 1
        return {'queue_running':[],'queue_pending':[[i,f'existing-{i}'] for i in range(8)]}
    backend.queue=full_queue
    response=submit(client,files=[('input_reference',('a.png',png(),'image/png'))])
    assert response.status_code == 429
    assert backend.queue_calls == 1 and not backend.uploaded and backend.submitted_graph is None

@pytest.mark.parametrize('fault',['dynamic_prefix','dynamic_required','dynamic_type','dynamic_max','nested_codec','nested_required','malformed'])
def test_ref_runtime_dynamic_contract_fails_closed(api,fault):
    client, backend = api
    template=backend.schemas['MiniMaxH3ReferenceToVideo']['input']['optional']['ref_images'][1]['template']
    if fault == 'dynamic_prefix': template['prefix']='unexpected_'
    elif fault == 'dynamic_required': template['input']['required']['other']=['AUDIO']
    elif fault == 'dynamic_type': template['input']['required']['ref_image'][0]='AUDIO'
    elif fault == 'dynamic_max': template['max']=0
    elif fault == 'malformed': template['input']=None
    else:
        fmt=backend.schemas['SaveVideo']['input']['required']['format'][1]['options']
        selected=next(item for item in fmt if item['key']=='mp4')
        if fault == 'nested_codec': selected['inputs']['required']['codec'][1]['options']=[]
        else: selected['inputs']['required']['other']=['INT']
    response=submit(client,files=[('images',('a.png',png(),'image/png'))],semantic_references='[{"image_index":0}]')
    assert response.status_code == 503, response.text
    assert backend.queue_calls == 0 and not backend.uploaded

@pytest.mark.parametrize('result',[('../bad.png',''),('a.png','unsafe/path'),('bad.webp',''),(None,'')])
def test_unsafe_upstream_input_names_never_submit(api,result):
    client,backend=api
    async def unsafe(*args): return result
    backend.upload_image=unsafe
    assert submit(client,files=[('input_reference',('a.png',png(),'image/png'))]).status_code == 502
    assert backend.submitted_graph is None

def test_timing_survives_new_app_and_old_ltx_receipts(api,tmp_path):
    from easel.video_jobs import make_video_id
    client,backend=api
    response=submit(client)
    assert response.status_code == 200
    receipt=response.json()
    assert receipt['frames']==124
    settings=dataclasses.replace(Settings.from_env({}),comfy_video_url='http://video',api_key='secret',image_job_dir=str(tmp_path/'restart'))
    backend.history={'status':{'completed':True},'outputs':{'save':{'videos':[{'filename':'output.mp4','type':'output'}]}}}
    with TestClient(create_app(settings=settings,comfy=backend,comfy_video=backend)) as fresh:
        fresh.headers['Authorization']='Bearer secret'
        assert fresh.get('/v1/videos/'+receipt['id']).json()['frames']==124
        assert fresh.get('/v1/videos/queue/'+receipt['id']).json()['frames']==124
        assert fresh.get('/v1/videos/'+receipt['id']+'/content').content==b'mp4-output'
        legacy=make_video_id('duration-test-prompt-id','ltx-2.5')
        body=fresh.get('/v1/videos/'+legacy).json()
        assert body['model']=='ltx-2.5' and 'frames' not in body
        assert fresh.get('/v1/videos/'+legacy+'/content').content==b'mp4-output'

def test_capabilities_remain_default_ltx(api):
    client,backend=api
    # H3's fake has no LTX guide schemas, and still preserves LTX discovery.
    body=client.get('/v1/videos/capabilities').json()
    assert body['model']=='ltx-2.5' and body['seconds']['default']==4
    backend.schemas.pop('LoraLoaderModelOnly')
    h3=client.get('/v1/videos/capabilities?model=minimax-h3').json()
    assert not h3['profiles']['fl8']['available']
    assert h3['profiles']['ref20']['available']
    assert h3['profiles']['temporal_ref20']['available']
    assert backend.queue_calls==0 and not backend.uploaded

def test_temporal_multiple_groups_keep_order_and_original_latent(api):
    client,backend=api
    files=[('images',(f'{i}.png',png(),'image/png')) for i in range(7)]
    metadata=[{'frame_index':80,'image_indices':[6]}, {'frame_index':0,'image_indices':[0,1,2,3,4]}, {'frame_index':40,'image_indices':[5]}]
    response=submit(client,files=files,temporal_groups=json.dumps(metadata))
    assert response.status_code==200,response.text
    graph=backend.submitted_graph
    guides=[(key,n['inputs']) for key,n in graph.items() if n['class_type']=='MiniMaxH3AddGuide']
    av=next(key for key,n in graph.items() if n['class_type']=='MiniMaxH3ReferenceToVideo')
    assert [item['frame_idx'] for _,item in guides]==[0,40,80]
    assert all(item['latent']==[av,1] for _,item in guides)
    assert guides[1][1]['positive']==[guides[0][0],0]
    assert guides[2][1]['positive']==[guides[1][0],0]

@pytest.mark.parametrize('fields', [
 {'semantic_references':'[{"image_index":0,"image_index":1}]'},
 {'temporal_groups':'[{"frame_index":0,"image_indices":[0,1,2,3,4]},{"frame_index":4,"image_indices":[5]}]'},
 {'temporal_groups':'[{"frame_index":0,"image_indices":[0,1,2,3,4]},{"frame_index":20,"image_indices":[4]}]'},
 {'semantic_references':'[{"image_index":0}]','temporal_groups':'[{"frame_index":0,"image_indices":[0]}]'},
 {'semantic_references':'[{"image_index":0},{"image_index":1},{"image_index":2}]'},
])
def test_metadata_conflicts_and_duplicate_json_keys(api,fields):
    client,backend=api
    files=[('images',(f'{i}.png',png(),'image/png')) for i in range(6)]
    assert submit(client,files=files,**fields).status_code==400
    assert not backend.info_calls and backend.queue_calls==0

@pytest.mark.parametrize('options', [dict(semantic_references='[{"image_index":0}]'),dict(temporal_groups='[{"frame_index":0,"image_indices":[0]}]')])
def test_first_frame_conflicts_with_ref20_controls(api,options):
    client,backend=api
    files=[('input_reference',('a.png',png(),'image/png')),('images',('b.png',png(),'image/png'))]
    assert submit(client,files=files,**options).status_code==400
    assert not backend.info_calls and backend.queue_calls==0

def test_maximum_bounded_temporal_graph(api):
    client,backend=api
    count=119
    files=[('images',(f'{i}.png',png(),'image/png')) for i in range(count)]
    groups=[{'frame_index':39*g,'image_indices':list(range(2+39*g,2+39*(g+1)))} for g in range(3)]
    response=submit(client,files=files,frames='362',semantic_references='[{"image_index":0},{"image_index":1}]',temporal_groups=json.dumps(groups))
    assert response.status_code==200,response.text
    assert len(backend.uploaded)==119 and len(backend.submitted_graph)<=256

@pytest.mark.parametrize('failure',['upload','submit','discovery'])
def test_upstream_failure_is_sanitized_and_never_retried(api,failure):
    from easel.comfy_client import ComfyError
    client,backend=api
    calls=[]
    async def fail(*args):
        calls.append(1)
        raise ComfyError('secret /private/file.png token=hidden')
    if failure=='upload': backend.upload_image=fail
    elif failure=='submit': backend.submit=fail
    else: backend.object_info=fail
    response=submit(client,files=[('input_reference',('a.png',png(),'image/png'))])
    assert response.status_code==502
    assert len(calls)==1 and 'private' not in response.text and 'hidden' not in response.text
    assert backend.submitted_graph is None

def jpeg_orientation(orientation):
    out=io.BytesIO()
    image=Image.new('RGB',(8,16),'red')
    exif=Image.Exif()
    exif[274]=orientation
    image.save(out,format='JPEG',exif=exif)
    return out.getvalue()

def test_temporal_dimensions_account_for_exif_rotation(api):
    client,backend=api
    files=[('images',(f'{i}.jpg',jpeg_orientation(1 if i==0 else 6),'image/jpeg')) for i in range(5)]
    response=submit(client,files=files,temporal_groups='[{"frame_index":0,"image_indices":[0,1,2,3,4]}]')
    assert response.status_code==400
    assert not backend.info_calls and backend.queue_calls==0 and not backend.uploaded

def test_h3_lora_catalog_has_no_client_selectable_adapters(api):
    client,backend=api
    response=client.get('/v1/videos/loras?model=minimax-h3')
    assert response.status_code==200
    assert response.json()=={'object':'list','model':'minimax-h3','data':[]}
    assert not backend.info_calls
    assert client.get('/v1/videos/loras?model=unknown').status_code==400
    assert not backend.info_calls

def test_capability_discovery_failure_is_sanitized(api):
    from easel.comfy_client import ComfyError
    client,backend=api
    async def fail(*args): raise ComfyError('secret /private/file token=hidden')
    backend.object_info=fail
    response=client.get('/v1/videos/capabilities?model=minimax-h3')
    assert response.status_code==502
    assert 'private' not in response.text and 'hidden' not in response.text
    assert backend.queue_calls==0

@pytest.mark.parametrize('field',['output_is_list','is_input_list'])
def test_runtime_list_semantics_must_match_selected_native_graph(api,field):
    client,backend=api
    backend.schemas['MiniMaxH3ImageToVideo'][field]=[True,False] if field=='output_is_list' else True
    assert submit(client).status_code==503
    assert backend.queue_calls==0 and not backend.uploaded


@pytest.mark.parametrize('response_fields', INVALID_UPLOAD_RESPONSES)
def test_real_comfy_upload_transport_malformed_success_returns_stable_h3_502(tmp_path,response_fields):
    import httpx
    from easel.comfy_client import ComfyClient
    calls=[]
    def handler(request):
        calls.append((request.method,request.url.path))
        if request.url.path.startswith('/object_info/'):
            name=request.url.path.rsplit('/',1)[1]
            return httpx.Response(200,json={name:SCHEMAS[name]})
        if request.url.path=='/queue':
            return httpx.Response(200,json={'queue_running':[],'queue_pending':[]})
        if request.url.path=='/upload/image':
            assert request.method=='POST'
            assert b'filename="h3_' in request.content
            return httpx.Response(200,**response_fields)
        raise AssertionError(f'unexpected mutation: {request.method} {request.url.path}')
    http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend=ComfyClient('http://comfy',http)
    settings=dataclasses.replace(Settings.from_env({}),comfy_video_url='http://video',api_key='secret',image_job_dir=str(tmp_path))
    with TestClient(create_app(settings=settings,comfy=backend,comfy_video=backend)) as client:
        client.headers['Authorization']='Bearer secret'
        response=submit(client,files=[('input_reference',('image.png',png(),'image/png'))])
    assert response.status_code==502
    assert response.json()['error']['code']=='upstream_upload_error'
    assert response.json()['error']['message']=='H3 still upload failed'
    assert 'private' not in response.text and 'diagnostic' not in response.text
    assert [path for method,path in calls if method=='POST']==['/upload/image']
