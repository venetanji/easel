"""Bounded local audio decoding and managed ComfyUI video inputs."""
from __future__ import annotations

import asyncio
import io
import json
import math
import re
import shutil
import tempfile
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
from starlette.datastructures import UploadFile

from .comfy_client import ComfyError
from .errors import APIError

MAX_AUDIO_BYTES = 32 * 1024 * 1024
SAMPLE_RATE = 48_000
DECODE_TIMEOUT = 15
AUDIO_FORMATS = {
    'audio/wav': 'wav', 'audio/x-wav': 'wav', 'audio/wave': 'wav',
    'audio/mpeg': 'mp3', 'audio/mp3': 'mp3',
    'audio/flac': 'flac', 'audio/x-flac': 'flac',
    'audio/ogg': 'ogg', 'application/ogg': 'ogg',
    'audio/mp4': 'mov', 'audio/x-m4a': 'mov',
}
AUDIO_MIME_TYPES = (*AUDIO_FORMATS, 'application/octet-stream')


@dataclass(frozen=True)
class SourceAudio:
    name: str
    data: bytes


def audio_tools_available() -> bool:
    return bool(shutil.which('ffprobe') and shutil.which('ffmpeg'))


def upload_selector_available(port: object, upload_flag: str) -> bool:
    if not isinstance(port, list) or len(port) < 2 or not isinstance(port[1], dict):
        return False
    kind, options = port[0], port[1]
    if options.get(upload_flag) is not True or options.get('multiselect', False) is not False:
        return False
    choices = kind if isinstance(kind, list) else options.get('options') if kind == 'COMBO' else None
    return isinstance(choices, list) and all(isinstance(choice, str) for choice in choices)


async def _run(*arguments: str) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        raise APIError(503, 'source_audio requires ffmpeg and ffprobe',
                       code='audio_decoder_unavailable', param='source_audio') from None
    try:
        output, _ = await asyncio.wait_for(process.communicate(), DECODE_TIMEOUT)
    except BaseException as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.communicate()
        if isinstance(exc, TimeoutError):
            raise APIError(400, 'source_audio decoding timed out', param='source_audio') from None
        raise
    if process.returncode:
        raise APIError(400, 'source_audio could not be decoded safely', param='source_audio')
    return output


async def validate_source_audio(upload: UploadFile | None, duration: float) -> SourceAudio | None:
    if upload is None:
        return None
    if not isinstance(upload, UploadFile) or upload.content_type not in AUDIO_MIME_TYPES:
        raise APIError(400, 'source_audio must be a WAV, MP3, FLAC, Ogg or M4A audio upload',
                       param='source_audio')
    if not math.isfinite(duration) or not 0 < duration <= 16:
        raise ValueError('source audio duration must be bounded by the video request')
    try:
        data = await upload.read(MAX_AUDIO_BYTES + 1)
    finally:
        await upload.seek(0)
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise APIError(400, 'source_audio must be nonempty and at most 32 MiB', param='source_audio')
    sample_count = round(duration * SAMPLE_RATE)
    with tempfile.TemporaryDirectory(prefix='easel-audio-') as directory:
        filename = str(Path(directory) / 'source')
        Path(filename).write_bytes(data)
        restrictions = ('-protocol_whitelist', 'file,pipe', '-format_whitelist', 'wav,mp3,flac,ogg,mov')
        probe = await _run('ffprobe', '-v', 'error', *restrictions, '-select_streams', 'a:0',
                           '-show_entries', 'stream=sample_rate,channels:format=format_name',
                           '-of', 'json', filename)
        try:
            info = json.loads(probe)
            stream = info['streams'][0]
            formats = info['format']['format_name'].split(',')
            expected = AUDIO_FORMATS.get(upload.content_type)
            if (not 1 <= int(stream['channels']) <= 8 or
                    not 1 <= int(stream['sample_rate']) <= 192_000 or
                    expected is not None and expected not in formats):
                raise ValueError('unsupported audio stream')
        except (ValueError, TypeError, KeyError, IndexError):
            raise APIError(400, 'source_audio bytes or stream do not match a supported audio type',
                           param='source_audio') from None
        pcm = await _run('ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', *restrictions,
                         '-i', filename, '-map', '0:a:0', '-vn', '-sn', '-dn',
                         '-t', str(sample_count / SAMPLE_RATE), '-ar', str(SAMPLE_RATE),
                         '-ac', '2', '-f', 's16le', 'pipe:1')
    if len(pcm) != sample_count * 4:
        raise APIError(400, f'source_audio must cover the rendered video ({duration:.6f} seconds)',
                       code='audio_too_short', param='source_audio')
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return SourceAudio('easel_audio_' + uuid.uuid4().hex + '.wav', output.getvalue())


async def upload_source_audio(comfy, audio: SourceAudio) -> str:
    try:
        name, subfolder = await comfy.upload_image(audio.data, audio.name, 'audio/wav')
    except (ComfyError, httpx.HTTPError):
        raise APIError(502, 'source_audio upload failed', type='api_error',
                       code='upstream_upload_error', param='source_audio') from None
    if (subfolder or not isinstance(name, str) or len(name) > 200 or
            re.fullmatch(r'[A-Za-z0-9_-]+\.wav', name) is None):
        raise APIError(502, 'upstream returned an unsafe audio input name', type='api_error',
                       code='upstream_upload_error', param='source_audio')
    return name


async def _audio_nodes_available(comfy, *, width: int, height: int, duration: float) -> bool:
    contracts = {
        'LoadAudio': ({'audio': None}, ['AUDIO']),
        'TrimAudioDuration': ({'audio': 'AUDIO', 'start_index': 'FLOAT', 'duration': 'FLOAT'}, ['AUDIO']),
        'LTXVAudioVAEEncode': ({'audio': 'AUDIO', 'audio_vae': 'VAE'}, ['LATENT']),
        'SolidMask': ({'value': 'FLOAT', 'width': 'INT', 'height': 'INT'}, ['MASK']),
        'SetLatentNoiseMask': ({'samples': 'LATENT', 'mask': 'MASK'}, ['LATENT']),
    }
    values = {'TrimAudioDuration': {'start_index': 0.0, 'duration': duration},
              'SolidMask': {'value': 0.0, 'width': width // 2, 'height': height // 2}}
    for name, (fields, outputs) in contracts.items():
        info = await comfy.object_info(name)
        schema = info.get(name) if isinstance(info, dict) else None
        if not isinstance(schema, dict) or schema.get('output') != outputs:
            return False
        if (schema.get('is_input_list', False) is not False or
                schema.get('output_is_list', [False] * len(outputs)) != [False] * len(outputs)):
            return False
        inputs = schema.get('input', {})
        if not isinstance(inputs, dict):
            return False
        required = inputs.get('required', {})
        optional = inputs.get('optional', {})
        if not isinstance(required, dict) or not isinstance(optional, dict) or set(required) - set(fields):
            return False
        declared = {**optional, **required}
        for field, kind in fields.items():
            port = declared.get(field)
            if not isinstance(port, list) or not port:
                return False
            if kind is None:
                if not upload_selector_available(port, 'audio_upload'):
                    return False
            elif port[0] != kind:
                return False
            if field in values.get(name, {}):
                bounds = port[1] if len(port) > 1 else {}
                if not isinstance(bounds, dict):
                    return False
                for bound in ('min', 'max'):
                    if bound in bounds and (type(bounds[bound]) not in (int, float) or not math.isfinite(bounds[bound])):
                        return False
                value = values[name][field]
                if not bounds.get('min', value) <= value <= bounds.get('max', value):
                    return False
    return True


async def audio_nodes_available(comfy, *, width: int = 1280, height: int = 720,
                                duration: float = 289 / 24) -> bool:
    try:
        return await _audio_nodes_available(comfy, width=width, height=height, duration=duration)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return False


def source_audio_capabilities(available: bool) -> dict:
    return {'supported': True, 'available': available and audio_tools_available(),
            'field': 'source_audio', 'mime_types': list(AUDIO_MIME_TYPES),
            'max_bytes': MAX_AUDIO_BYTES, 'start_seconds': 0,
            'duration_policy': 'must_cover_rendered_video; longer_audio_is_trimmed',
            'conditioning': True, 'output': 'trimmed_source_audio',
            'normalization': '48000 Hz stereo PCM WAV',
            'validation': 'graph_contract_tested', 'gpu_executed': False}
