"""Bounded H3 multipart admission and read-only native runtime discovery."""
from __future__ import annotations

import json
import math
import re
import uuid
import warnings
from dataclasses import dataclass

from PIL import Image
from starlette.datastructures import UploadFile

from . import h3_graph as h3
from .errors import APIError
from .video_sizing import VideoLimits, VideoSizingError, parse_video_size

MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 64_000_000
MAX_IMAGE_PIXELS = 32_000_000
MAX_IMAGES = 2 + 3 * 39
FIELDS = {'model', 'prompt', 'frames', 'seed', 'size', 'input_reference',
          'images', 'semantic_references', 'temporal_groups'}
OUTPUTS = {
    'UNETLoader': ['MODEL'], 'CLIPLoader': ['CLIP'], 'VAELoader': ['VAE'],
    'LoraLoaderModelOnly': ['MODEL'], 'MiniMaxH3SigmaShift': ['MODEL'],
    'MiniMaxH3ImageToVideo': ['CONDITIONING', 'LATENT'],
    'MiniMaxH3ReferenceToVideo': ['CONDITIONING', 'LATENT'],
    'MiniMaxH3AddGuide': ['CONDITIONING'], 'LoadImage': ['IMAGE', 'MASK'],
    'ImageBatch': ['IMAGE'], 'BasicGuider': ['GUIDER'], 'RandomNoise': ['NOISE'],
    'KSamplerSelect': ['SAMPLER'], 'BasicScheduler': ['SIGMAS'],
    'SamplerCustomAdvanced': ['LATENT', 'LATENT'], 'VAEDecode': ['IMAGE'],
    'VAEDecodeAudio': ['AUDIO'], 'CreateVideo': ['VIDEO'], 'SaveVideo': ['VIDEO'],
}


@dataclass(frozen=True)
class SemanticReference:
    image_index: int


@dataclass(frozen=True)
class TemporalGroup:
    frame_index: int
    image_indices: tuple[int, ...]


@dataclass(frozen=True)
class H3Request:
    prompt: str
    frames: int
    seed: int
    width: int
    height: int
    uploads: tuple[UploadFile, ...]
    names: tuple[str, ...]
    references: tuple[SemanticReference, ...]
    groups: tuple[TemporalGroup, ...]
    first_frame: bool

    def graph(self, names: tuple[str, ...] | None = None) -> dict:
        names = self.names if names is None else names
        options = dict(frames=self.frames, seed=self.seed, width=self.width, height=self.height)
        refs = tuple(names[ref.image_index] for ref in self.references)
        if self.groups:
            return h3.build_temporal_guided_video(self.prompt, guides=[
                h3.TemporalGuide(tuple(names[i] for i in group.image_indices), group.frame_index)
                for group in self.groups], reference_filenames=refs, **options)
        if refs:
            return h3.build_r2v(self.prompt, reference_filenames=refs, **options)
        if self.first_frame:
            return h3.build_i2v(self.prompt, image_filename=names[0], **options)
        return h3.build_t2v(self.prompt, **options)


def invalid(message: str, param: str) -> None:
    raise APIError(400, message, param=param)


def integer(form, field: str, default: int, minimum: int, maximum: int) -> int:
    value = form.get(field)
    if value is None:
        return default
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{1,20}', value):
        invalid(f'{field} must be an unsigned decimal integer', field)
    result = int(value)
    if not minimum <= result <= maximum:
        invalid(f'{field} must be in {minimum}..{maximum}', field)
    return result


def metadata(form, field: str, maximum: int) -> list:
    value = form.get(field)
    if value is None:
        return []
    if not isinstance(value, str) or len(value) > 8192:
        invalid(f'{field} must be bounded JSON metadata', field)
    # Repeated JSON keys are ambiguous even when their last value looks valid.
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = item
        return result
    try:
        items = json.loads(value, object_pairs_hook=unique)
    except (ValueError, RecursionError):
        invalid(f'{field} must be a JSON array of typed objects', field)
    if not isinstance(items, list) or not 1 <= len(items) <= maximum:
        invalid(f'{field} must contain 1..{maximum} entries', field)
    return items


async def admit(form, *, limits: VideoLimits = VideoLimits()) -> H3Request:
    for key in form:
        if key not in FIELDS:
            invalid(f'unsupported H3 parameter: {key}', key)
        values = form.getlist(key)
        if key != 'images' and len(values) != 1:
            invalid(f'{key} must occur only once', key)
        if key in ('images', 'input_reference'):
            if any(not isinstance(value, UploadFile) for value in values):
                invalid(f'{key} must contain still image uploads', key)
        elif any(not isinstance(value, str) or not value.strip() for value in values):
            invalid(f'{key} must contain a nonempty string', key)
    prompt = form['prompt'].strip()
    if '\x00' in prompt:
        invalid('prompt must not contain NUL', 'prompt')
    frames = integer(form, 'frames', 124, 124, 362)
    if (frames - 5) % 17:
        invalid('frames must lie on the 17k+5 grid', 'frames')
    seed = integer(form, 'seed', 0, 0, 2**64-1)
    try:
        width, height = parse_video_size(form.get('size'), model=h3.VIDEO_MODEL_ID, limits=limits)
        limits.check_frames(h3.VIDEO_MODEL_ID, width, height, frames)
    except VideoSizingError as exc:
        raise APIError(400, str(exc), param=exc.param, code=exc.code, details=exc.details) from None
    refs = []
    for item in metadata(form, 'semantic_references', 2):
        if not isinstance(item, dict) or set(item) != {'image_index'} or type(item['image_index']) is not int:
            invalid('semantic_references needs image_index objects only', 'semantic_references')
        refs.append(SemanticReference(item['image_index']))
    groups = []
    for item in metadata(form, 'temporal_groups', 3):
        if not isinstance(item, dict) or set(item) != {'frame_index', 'image_indices'}:
            invalid('temporal_groups needs frame_index/image_indices objects only', 'temporal_groups')
        indices, frame = item['image_indices'], item['frame_index']
        if (not isinstance(indices, list) or len(indices) not in (1,5,22,39) or
                any(type(index) is not int for index in indices) or type(frame) is not int or
                not 0 <= frame <= frames-len(indices)):
            invalid('invalid temporal group count, indices, or frame bounds', 'temporal_groups')
        groups.append(TemporalGroup(frame, tuple(indices)))
    groups.sort(key=lambda group: group.frame_index)
    for previous, current in zip(groups, groups[1:]):
        if previous.frame_index + len(previous.image_indices) > current.frame_index:
            invalid('temporal groups must not overlap', 'temporal_groups')
    first_frame = 'input_reference' in form
    uploads = tuple(form.getlist('images'))
    if first_frame:
        if uploads or refs or groups:
            invalid('input_reference cannot be combined with semantic or temporal conditioning', 'input_reference')
        uploads = (form['input_reference'],)
    else:
        indices = [ref.image_index for ref in refs] + [i for group in groups for i in group.image_indices]
        if (len(indices) != len(uploads) or len(set(indices)) != len(indices) or
                set(indices) != set(range(len(uploads)))):
            invalid('each images upload must be declared exactly once by a valid index', 'images')
    if len(uploads) > MAX_IMAGES:
        invalid('too many H3 still uploads', 'images')
    names, dimensions = [], []
    remaining_bytes, remaining_pixels = MAX_BYTES, MAX_PIXELS
    for upload in uploads:
        field = 'input_reference' if first_frame else 'images'
        if upload.content_type not in ('image/png', 'image/jpeg'):
            invalid('H3 accepts only PNG/JPEG still images', field)
        size = 0
        while True:
            chunk = await upload.read(min(1024*1024, remaining_bytes+1))
            if not chunk:
                break
            size += len(chunk)
            remaining_bytes -= len(chunk)
            if remaining_bytes < 0:
                invalid('H3 still uploads exceed 32 MiB combined', field)
        await upload.seek(0)
        if not size:
            invalid('still upload must not be empty', field)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(upload.file) as image:
                    if image.format not in ('PNG','JPEG') or image.get_format_mimetype() != upload.content_type:
                        invalid('still bytes do not match PNG/JPEG MIME type', field)
                    if getattr(image, 'n_frames', 1) != 1:
                        invalid('H3 uploads must contain exactly one still image', field)
                    pixels = image.width * image.height
                    remaining_pixels -= pixels
                    if pixels > MAX_IMAGE_PIXELS or remaining_pixels < 0:
                        invalid('H3 images exceed 32 MP each or 64 MP combined', field)
                    image.load()
                    # Native LoadImage applies EXIF transpose before ImageBatch.
                    # Orientations 5..8 swap axes; compare those actual dimensions
                    # without allocating a second full decoded pixel buffer.
                    orientation = image.getexif().get(274)
                    dimensions.append((image.height, image.width) if orientation in (5, 6, 7, 8) else image.size)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            invalid('still image could not be decoded safely', field)
        finally:
            await upload.seek(0)
        names.append('h3_'+uuid.uuid4().hex+('.png' if upload.content_type == 'image/png' else '.jpg'))
    for group in groups:
        if len({dimensions[i] for i in group.image_indices}) != 1:
            invalid('all images in each temporal group must have matching dimensions', 'temporal_groups')
    return H3Request(prompt, frames, seed, width, height, uploads, tuple(names), tuple(refs), tuple(groups), first_frame)


def _schema_for(declared, field):
    if field.startswith('ref_images.ref_image_'):
        schema = declared.get('ref_images')
        if (not isinstance(schema, list) or len(schema) < 2 or schema[0] != 'COMFY_AUTOGROW_V3' or
                not isinstance(schema[1], dict)):
            return None
        template = schema[1].get('template')
        if (not isinstance(template, dict) or template.get('prefix') != 'ref_image_' or
                type(template.get('max')) is not int or template['max'] < 2 or
                template.get('min', 0) != 0):
            return None
        inputs = template.get('input', {})
        if not isinstance(inputs, dict):
            return None
        required = inputs.get('required', {})
        if not isinstance(required, dict) or set(required) != {'ref_image'}:
            return None
        return required['ref_image']
    return declared.get(field)


def _compatible(schema, value, graph, field, node_type) -> bool:
    if not isinstance(schema, list) or not schema:
        return False
    kind = schema[0]
    options = schema[1] if len(schema) > 1 else {}
    if not isinstance(options, dict):
        return False
    if isinstance(value, list):
        source, output = value
        return kind == OUTPUTS[graph[source]['class_type']][output]
    # Upload basenames do not yet appear in LoadImage's file chooser.
    if node_type == 'LoadImage' and field == 'image':
        return isinstance(kind, list) and options.get('image_upload') is True
    if isinstance(kind, list):
        return value in kind
    if kind in ('COMBO', 'COMFY_DYNAMICCOMBO_V3'):
        choices = options.get('options')
        if not isinstance(choices, list):
            return False
        if kind == 'COMFY_DYNAMICCOMBO_V3':
            choices = [item.get('key') for item in choices if isinstance(item, dict)]
        return value in choices
    if kind == 'STRING':
        return isinstance(value, str)
    if kind not in ('INT', 'FLOAT') or type(value) not in (int, float) or not math.isfinite(value):
        return False
    if kind == 'INT' and type(value) is not int:
        return False
    for bound in ('min','max'):
        if bound in options and (type(options[bound]) not in (int,float) or not math.isfinite(options[bound])):
            return False
    return options.get('min', value) <= value <= options.get('max', value)


async def _runtime_available(comfy, graph: dict) -> bool:
    """Check only selected graph nodes/curated assets without loading or mutating."""
    schemas = {}
    for node in graph.values():
        name = node['class_type']
        if name not in schemas:
            info = await comfy.object_info(name)
            schema = info.get(name) if isinstance(info, dict) else None
            if not isinstance(schema, dict) or schema.get('output') != OUTPUTS[name]:
                return False
            if (schema.get('is_input_list', False) is not False or
                    schema.get('output_is_list', [False] * len(OUTPUTS[name])) != [False] * len(OUTPUTS[name])):
                return False
            inputs = schema.get('input')
            if not isinstance(inputs, dict):
                return False
            required, optional = inputs.get('required', {}), inputs.get('optional', {})
            if not isinstance(required, dict) or not isinstance(optional, dict):
                return False
            schemas[name] = required, {**optional, **required}
        required, declared = schemas[name]
        values = node['inputs']
        if set(required) - set(values):
            return False
        for field, value in values.items():
            if not _compatible(_schema_for(declared, field), value, graph, field, name):
                return False
        if name == 'SaveVideo':
            # V3 expands codec from the selected container; validate that branch,
            # including new required nested inputs, rather than hidden defaults.
            fmt = declared['format']
            if fmt[0] == 'COMFY_DYNAMICCOMBO_V3':
                selected = next(item for item in fmt[1]['options'] if item.get('key') == 'mp4')
                nested = selected.get('inputs', {}).get('required', {})
                if set(nested) != {'codec'} or not _compatible(nested['codec'], 'h264', graph, 'codec', name):
                    return False
                codec = next(item for item in nested['codec'][1]['options'] if item.get('key') == 'h264')
                if codec.get('inputs', {}).get('required', {}):
                    return False
    return True


async def runtime_available(comfy, graph: dict) -> bool:
    try:
        return await _runtime_available(comfy, graph)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration):
        # Malformed remote JSON is an incompatible contract, never a decoder traceback.
        return False


async def capabilities(comfy, *, limits: VideoLimits = VideoLimits()) -> dict:
    common = dict(frames=362, seed=2**64-1)
    profiles = {
        'fl8': h3.build_i2v('discovery', image_filename='h3_probe.png', **common),
        'ref20': h3.build_r2v('discovery', reference_filenames=['h3_probe.png','h3_probe2.png'], **common),
        'temporal_ref20': h3.build_temporal_guided_video('discovery', guides=[h3.TemporalGuide(tuple(f'h3_probe{i}.png' for i in range(39)), 323)], reference_filenames=['h3_ref.png','h3_ref2.png'], **common),
    }
    result = {}
    for name, graph in profiles.items():
        result[name] = {'supported': True, 'available': await runtime_available(comfy, graph),
                        'validation': 'graph_contract_tested', 'gpu_executed': False, 'visually_reviewed': False}
    return {'object':'video.capabilities','schema_version':1,'model':h3.VIDEO_MODEL_ID,
            'fps':24,'frames':{'min':124,'max':362,'default':124,'step':17,'offset':5},
            **limits.discovery(h3.VIDEO_MODEL_ID),'default_size':'864x480',
            'seed':{'min':'0','max':str(2**64-1),'encoding':'decimal_string'},
            'profiles':result,'semantic_references':{'max_count':2,'fields':['image_index']},
            'temporal_groups':{'max_count':3,'image_counts':[1,5,22,39], 'fields':['frame_index','image_indices']},
            'uploads':{'field':'images','mime_types':['image/png','image/jpeg'],'max_total_bytes':MAX_BYTES,
                       'max_total_pixels':MAX_PIXELS,'max_image_pixels':MAX_IMAGE_PIXELS,'max_count':MAX_IMAGES},
            'unsupported':['seconds','camera_lora','loras','motion_speed','source_audio','control_video','negative_prompt','cfg','custom_sampling']}
