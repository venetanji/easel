"""Single curated prompt-guide registry; reading a guide never calls an upstream."""
from __future__ import annotations

from copy import deepcopy
import json

from .errors import APIError

MAX_GUIDE_BYTES = 16 * 1024
_QWEN = 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/README.md'
_H3_BASE = 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/5d9b308a59ab12e67147f191e184baf704185bd1/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md'
_H3_NATIVE = 'https://github.com/Comfy-Org/ComfyUI/blob/b5cc8830279eae909a59de030af1e50761c36751/comfy_extras/nodes_minimax_h3.py#L145-L161'
_H3_REF = 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/5d9b308a59ab12e67147f191e184baf704185bd1/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md'
_AUDIO = '''Easel source audio: Check current exact-model video capabilities before supplying source_audio. When available, the upload conditions video and supplies its soundtrack: audio starts at zero, is normalized to 48 kHz stereo PCM WAV and trimmed to the rendered duration. Short audio is rejected, never padded. Rendered duration is frames/24 for H3 and (seconds*24+1)/24 for LTX. This is separate from Suno generation; there are no selectable copy/reference modes. Without an upload, the generated-sound guidance applies.'''
_DIALOGUE = '''Official paraphrase: For dialogue or singing use <d>[Language] ...</d>. Keep speaker identity, stable (S1)/(S2) ID, action and delivery outside the delimiters; inside include only the language tag and requested words, preserving punctuation and original language. Reuse speaker IDs across shots. Use N/A for absent score; overall_soundscape: N/A requests complete silence.'''
_BOUNDARY = 'Guidance cannot enable an operation: discover current capabilities first. Examples are authored illustrations, not validated generations. Reviewed 2026-10-12.'

_GUIDES = {
    ('moss-image-base', 'generation'): {
        'text': '''Easel recommendation for MOSS-Image-Base text-to-image: Describe the subject, composition, environment, lighting and style. Quote any visible text exactly. Upstream also accepts tag sequences and JSON layout descriptions; use its user guide for the layout grammar rather than inventing field names.
Sampling follows the official native ComfyUI example: 1024x1024, 50 steps, CFG 3, shift 5, Euler / sgm_uniform, and negative-branch skipping at single layer 22. Easel uses FP8 transformer weights and CPU text encoding for the 12 GB image GPU. CPU text encoding still requires substantial host RAM; start with one image. Easel serves generation only: no reference images, edits, variations or video-server routing for this model.
Authored example: A small red ceramic teapot on a walnut table beside a rainy window, soft overcast daylight, gentle steam, detailed glaze, editorial product photograph.
Reviewed 2026-10-17. Examples illustrate prompting; they are not quality guarantees.''',
        'sources': [
            'https://github.com/OpenMOSS/MOSS-Image/blob/c1355e6b1986dfe8fba3c0455783a05899212342/USERGUIDE.md',
            'https://github.com/OpenMOSS/MOSS-Image/blob/c1355e6b1986dfe8fba3c0455783a05899212342/comfyui/moss-image/example_workflows/moss-image-api.json',
        ],
    },
    ('qwen-image-2.1', 'generation'): {
        'text': '''Easel recommendation for native Qwen-Image-2.1 generation (no input image): Describe the subject, arrangement, environment, lighting and visual style. Specify visible text exactly. For transparency explicitly request RGBA, an alpha channel and a transparent background.
Official basis: The model card demonstrates descriptive generation, quoted text and transparency; it does not prescribe a universal named-field layout.
Authored example: A small blue ceramic teapot centered in a square composition, soft studio light, subtle glaze texture. RGBA output with an alpha channel and a transparent background.
''' + _BOUNDARY,
        'sources': [_QWEN],
    },
    ('qwen-image-2.1', 'editing'): {
        'text': '''Easel recommendation for native Qwen-Image-2.1 image-conditioned editing: Name the change and the object or region it affects. State what must stay unchanged. Assign each supplied image a clear role using its input order; reference-order advice applies only when multiple images are supplied. Give exact replacement text for text edits. Request transparency explicitly when needed.
Official basis: The model card documents identity-preserving edits, local annotations and up to ten references. The change/target/preserve structure and numbered roles are Easel recommendations, not upstream grammar. An implementation ceiling is not an upstream quality claim; upstream mask support does not establish an Easel mask field. Generated PNG alpha still needs verification.
Authored example: Use image 1 as the base scene. Replace its mug with the mug from image 2, matching perspective and light. Keep the hands, table and background unchanged.
''' + _BOUNDARY,
        'sources': [_QWEN],
    },
    ('minimax-h3', 'base'): {
        'text': '''Official paraphrase for H3 FL8 text-to-video or first-frame image-to-video: Use these fields in order:
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
Describe composition, subjects, actions and camera movement along the timeline. Include camera speed and amplitude when useful. Separate environmental sound from audience-only music. Keep the opening consistent with an actual supplied first frame. Only with that first frame, precede the fields with this alignment line and a blank line:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
Later shots use increasing cut times within the clip duration. Do not infer last-frame or first/last-frame support.
''' + _DIALOGUE + '''
Authored dialogue fragment: The adult guide with a warm alto voice (S1) says softly: <d>[English] Follow the blue lantern.</d>
Easel recommendation: Fit actions and dialogue to authoritative frames/FPS duration; for a brief clip prefer one readable action unless cuts are needed.
Easel preprocessing (not prompt syntax): input_reference is stretched to the effective output dimensions, including with source_audio. A different source aspect ratio can distort subjects. There is no automatic padding and no selectable fit mode; Easel does not derive video size from the reference. Before creating an anchor, resolve the requested size or default_size using capabilities size_presets width/height and size_constraints.aliases, then prepare the anchor at that aspect ratio. For an existing reference, explain any mismatch before submission; do not silently change its framing or the requested video size. This describes first-image FL8 conditioning only, not REF20 semantic references or temporal groups.
''' + _AUDIO + '\n' + _BOUNDARY,
        'sources': [_H3_BASE, _H3_NATIVE],
    },
    ('minimax-h3', 'reference'): {
        'text': '''Official paraphrase for supported H3 REF20 semantic references: Use six fields in order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
Define reusable subjects and their source images. Keep labels consistent. Explain retained and changed attributes, then how referenced content participates in each shot. Subject labels denote reusable visual content; Picture labels identify source images or concrete composition/frame anchors. Identity-only sources need no separate Picture definition. Retention distinguishes full preservation, partial preservation, attribute transfer and weak reference. Upstream normally recommends a 350–500-word generation description; this help snippet does not impose a short final-prompt limit.
''' + _DIALOGUE + '''
In detailed_description retain a speaking reference's visual label and independent speaker ID as <Subject N> (Sx).
Authored dialogue fragment (Subject 1 already defined from a supplied image): <Subject 1> (S1), the adult guide with a warm alto voice, says softly: <d>[English] Follow the blue lantern.</d>
Easel recommendations: Build numbering from actual request mapping. Semantic references are not timeline anchors. Experimental temporal groups need a separately verified mapping; do not assume temporal uploads are semantic Pictures or infer first/last-frame claims from identity references. Retention expresses intent, not guaranteed pixel fidelity.
''' + _AUDIO + '\n' + _BOUNDARY,
        'sources': [_H3_BASE, _H3_REF],
    },
}


def get_model_guide(model: str, mode: str) -> dict:
    """Return a fresh version-1 envelope, or 404 for an unreviewed pair."""
    entry = _GUIDES.get((model, mode))
    if entry is None:
        raise APIError(404, 'No reviewed guide for this model and mode',
                       code='model_guide_unavailable', param='mode')
    result = {'version': 1, 'guide_id': f'{model}.{mode}', 'model': model,
              'mode': mode, **deepcopy(entry)}
    if len(json.dumps(result).encode('utf-8')) > MAX_GUIDE_BYTES:
        raise APIError(500, 'Curated model guide exceeds response limit', type='api_error',
                       code='model_guide_too_large')
    return result
