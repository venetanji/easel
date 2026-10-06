"""Public video-control contract and fail-closed request validation.

No graph, server filename, URL or client path is accepted here. Guide image
indices refer only to files supplied in the same multipart request.
"""
from __future__ import annotations

import json
import math
import warnings

from PIL import Image
from dataclasses import dataclass

from starlette.datastructures import UploadFile

from .errors import APIError
from .video_graph import FPS, VIDEO_MODEL_ID
from .video_sizing import VideoLimits
from .video_loras import MAX_VIDEO_LORAS

MAX_GUIDING_FRAMES = 8
MAX_VIDEO_UPLOAD_BYTES = 32 * 1024 * 1024
IMAGE_MIME_TYPES = ('image/png', 'image/jpeg', 'image/webp')
GUIDE_NODE_INPUTS = {
    'LTXVAddGuide': {'positive': 'CONDITIONING', 'negative': 'CONDITIONING',
        'vae': 'VAE', 'latent': 'LATENT', 'image': 'IMAGE', 'frame_idx': 'INT', 'strength': 'FLOAT'},
    'LTXVCropGuides': {'positive': 'CONDITIONING', 'negative': 'CONDITIONING', 'latent': 'LATENT'},
}
VIDEO_FORM_FIELDS = {
    'prompt', 'model', 'seconds', 'size', 'input_reference', 'camera_lora',
    'camera_lora_strength', 'loras', 'motion_speed', 'seed', 'lora_reference',
    'lora_reference_strength', 'guiding_frames', 'guiding_images',
}


@dataclass(frozen=True)
class GuidingFrame:
    image_index: int
    frame_index: int
    strength: float


def validate_video_form(form) -> None:
    for key in form:
        if key not in VIDEO_FORM_FIELDS:
            raise APIError(400, f'unsupported video parameter: {key}',
                           code='unsupported_parameter', param=key)
        if isinstance(form[key], str) and not form[key].strip():
            raise APIError(400, f'{key} must not be explicitly empty; omit it for the default', param=key)
        if key != 'guiding_images' and len(form.getlist(key)) != 1:
            raise APIError(400, f'{key} must occur only once', param=key)
        if key == 'guiding_images' and any(not isinstance(item, UploadFile) for item in form.getlist(key)):
            raise APIError(400, 'guiding_images must contain image file uploads', param=key)


def parse_guiding_frames(value: str | None, image_count: int, seconds: int) -> list[GuidingFrame]:
    if value is None:
        if image_count:
            raise APIError(400, 'guiding_images requires guiding_frames metadata', param='guiding_frames')
        return []
    message = 'guiding_frames must be a JSON array of image_index/frame_index/optional strength objects'
    if len(value) > 8192:
        raise APIError(400, 'guiding_frames JSON is too large', param='guiding_frames')
    try:
        requested = json.loads(value)
    except (ValueError, RecursionError):
        raise APIError(400, message, param='guiding_frames') from None
    if not isinstance(requested, list) or not 1 <= len(requested) <= MAX_GUIDING_FRAMES:
        raise APIError(400, f'guiding_frames must have 1 through {MAX_GUIDING_FRAMES} entries', param='guiding_frames')
    if image_count != len(requested):
        raise APIError(400, 'each guiding frame needs exactly one guiding_images upload', param='guiding_images')
    result = []
    for item in requested:
        if not isinstance(item, dict) or set(item) - {'image_index', 'frame_index', 'strength'}:
            raise APIError(400, message, param='guiding_frames')
        image, frame, strength = item.get('image_index'), item.get('frame_index'), item.get('strength', 1)
        if type(image) is not int or not 0 <= image < image_count:
            raise APIError(400, 'guide image_index must identify a supplied upload', param='guiding_frames')
        if type(frame) is not int or not 0 <= frame <= seconds * FPS:
            raise APIError(400, f'guide frame_index must be an integer from 0 through {seconds * FPS}', param='guiding_frames')
        if type(strength) not in (int, float) or not 0 <= strength <= 1 or not math.isfinite(strength):
            raise APIError(400, 'guide strength must be a finite number from 0 through 1', param='guiding_frames')
        result.append(GuidingFrame(image, frame, float(strength)))
    if len({item.image_index for item in result}) != len(result):
        raise APIError(400, 'guide image_index values must use each upload exactly once', param='guiding_frames')
    if len({item.frame_index for item in result}) != len(result):
        raise APIError(400, 'guide frame_index values must be unique', param='guiding_frames')
    return sorted(result, key=lambda item: item.frame_index)


async def validate_video_uploads(uploads: list[tuple[str, UploadFile]]) -> None:
    """Read only a bounded chunk at a time, rewind, and reject before upstream work."""
    remaining = MAX_VIDEO_UPLOAD_BYTES
    for field, upload in uploads:
        if upload.content_type not in IMAGE_MIME_TYPES:
            raise APIError(400, f'{field} must be a PNG, JPEG, or WebP image', param=field)
        size = 0
        while True:
            chunk = await upload.read(min(1024 * 1024, remaining + 1))
            if not chunk:
                break
            size += len(chunk)
            remaining -= len(chunk)
            if remaining < 0:
                raise APIError(400, 'video reference images exceed 32 MiB combined', param=field)
        await upload.seek(0)
        if not size:
            raise APIError(400, f'{field} must not be empty', param=field)
        if field == 'guiding_images':
            # AddGuide treats multi-frame IMAGE batches differently. Reject APNG/
            # animated WebP instead of silently taking a frame or changing timing.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(upload.file) as image:
                        if image.format not in ('PNG', 'JPEG', 'WEBP') or image.get_format_mimetype() != upload.content_type:
                            raise APIError(400, 'invalid guiding image: bytes do not match the declared image type',
                                           param=field)
                        if getattr(image, 'n_frames', 1) != 1:
                            raise APIError(400, 'invalid guiding image: guides must contain exactly one still image',
                                           param=field)
                        if image.width * image.height > 32_000_000:
                            raise APIError(400, 'invalid guiding image: guide images must not exceed 32 megapixels',
                                           param=field)
                        # verify() alone skips JPEG entropy decoding; load()
                        # rejects truncated/corrupt pixel data within the pixel cap.
                        image.load()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                # Decoder diagnostics may expose file objects, paths or addresses.
                raise APIError(400, 'invalid guiding image: image could not be decoded safely',
                               param=field) from None
            finally:
                await upload.seek(0)


async def guiding_nodes_available(comfy) -> bool:
    """Check runtime node shape; this is not GPU execution/visual validation."""
    for name, expected in GUIDE_NODE_INPUTS.items():
        info = await comfy.object_info(name)
        node = info.get(name) if isinstance(info, dict) else None
        if not isinstance(node, dict) or not isinstance(node.get('input'), dict):
            return False
        required = node['input'].get('required', {})
        optional = node['input'].get('optional', {})
        output = node.get('output')
        if not isinstance(required, dict) or not isinstance(optional, dict) or not isinstance(output, list):
            return False
        if set(required) - set(expected) or output[:3] != ['CONDITIONING', 'CONDITIONING', 'LATENT']:
            return False
        declared = {**optional, **required}
        for field, kind in expected.items():
            schema = declared.get(field)
            if not isinstance(schema, list) or not schema or schema[0] != kind:
                return False
            if field in ('frame_idx', 'strength'):
                bounds = schema[1] if len(schema) > 1 else {}
                if not isinstance(bounds, dict):
                    return False
                for bound in ('min', 'max'):
                    value = bounds.get(bound)
                    if bound in bounds and (type(value) not in (int, float) or not math.isfinite(value)):
                        return False
                maximum = 12 * FPS if field == 'frame_idx' else 1
                if bounds.get('min', 0) > 0 or bounds.get('max', maximum) < maximum:
                    return False
    return True


def video_capabilities(*, guides_available: bool, limits: VideoLimits = VideoLimits()) -> dict:
    return {
        'object': 'video.capabilities', 'schema_version': 1, 'model': VIDEO_MODEL_ID,
        'fps': FPS, 'seconds': {'min': 1, 'max': 12, 'default': 4},
        **limits.discovery(VIDEO_MODEL_ID), 'default_size': '1280x720',
        'seed': {'min': '0', 'max': str(2 ** 64 - 2), 'encoding': 'decimal_string'},
        'loras': {'max_count': MAX_VIDEO_LORAS, 'min_strength': 0, 'max_strength': 2,
                  'default_strength': 1, 'camera_default_strength': 0.8,
                  'catalog_path': '/v1/videos/loras'},
        'motion_speed': {'min': 0.025, 'max': 1, 'requires_lora': 'slow-motion'},
        'lora_reference_strength': {'min': 0, 'max': 1, 'default': 1, 'requires_lora': 'ingredients'},
        'uploads': {'mime_types': list(IMAGE_MIME_TYPES), 'max_total_bytes': MAX_VIDEO_UPLOAD_BYTES},
        'guiding_frames': {'supported': True, 'available': guides_available,
                           'validation': 'graph_contract_tested', 'max_count': MAX_GUIDING_FRAMES,
                           'frame_index_multiple': 1, 'min_strength': 0, 'max_strength': 1,
                           'default_strength': 1,
                           'exclusive_with': ['input_reference', 'lora_reference', 'ingredients']},
    }
