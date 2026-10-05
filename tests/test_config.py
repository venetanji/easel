"""Unit tests for the model table, size parsing, and settings."""
import pytest

from easel.config import MODELS, resolve_model, parse_size, UnknownModelError, Settings


# ---- model table ----

def test_resolve_known_models():
    qwen = resolve_model("qwen-image-2.1")
    assert qwen.unet == "qwen_image_2.1_int8_convrot.safetensors"
    assert qwen.clip == "qwen3vl_8b_int8_convrot.safetensors"
    assert qwen.vae == "qwen_image_2.1_vae_bf16.safetensors"
    assert qwen.backend == "qwen_image_2_1"
    assert qwen.default_steps == 25


def test_model_table_ids():
    assert set(MODELS) == {"qwen-image-2.1"}


@pytest.mark.parametrize("model", ["flux2-9b", "flux2-4b", "flux-2.5"])
def test_retired_flux_models_are_not_resolved(model):
    with pytest.raises(UnknownModelError):
        resolve_model(model)


def test_resolve_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        resolve_model("dall-e-3")


# ---- size parsing ----

def test_parse_size_none_auto_empty_return_none():
    assert parse_size(None) is None
    assert parse_size("auto") is None
    assert parse_size("") is None


def test_parse_size_exact():
    assert parse_size("1024x1024") == (1024, 1024)
    assert parse_size("512x768") == (512, 768)


def test_parse_size_case_and_whitespace_tolerant():
    assert parse_size(" 1024X1024 ") == (1024, 1024)


def test_parse_size_rounds_to_multiple_of_16():
    assert parse_size("1000x1000") == (1008, 1008)
    assert parse_size("300x300") == (304, 304)


def test_parse_size_clamps_range():
    assert parse_size("100x100") == (256, 256)
    assert parse_size("4000x4000") == (1536, 1536)


@pytest.mark.parametrize("bad", ["banana", "1024", "1024x", "x512", "1024*768", "-5x-5", "0x0"])
def test_parse_size_malformed_raises(bad):
    with pytest.raises(ValueError):
        parse_size(bad)


# ---- settings ----

def test_settings_defaults():
    s = Settings.from_env({})
    assert s.comfy_url.startswith("http")
    assert s.comfy_video_url is None
    assert s.api_key is None
    assert s.job_timeout == 600
    assert s.max_inflight == 1
    assert s.default_response_format == "b64_json"
    assert s.default_steps == 25
    assert s.variation_prompt
    assert s.prompt_separator == "|||"
    assert s.suno_url is None
    assert s.suno_api_token is None
    assert s.suno_timeout == 180


def test_suno_settings_are_optional_and_strip_blank_values():
    settings = Settings.from_env({"SUNO_URL": "  http://audio:8085/ ",
                                  "SUNO_REST_API_TOKEN": " secret ", "SUNO_TIMEOUT": "90"})
    assert settings.suno_url == "http://audio:8085/"
    assert settings.suno_api_token == "secret"
    assert settings.suno_timeout == 90
    assert Settings.from_env({"SUNO_URL": " ", "SUNO_REST_API_TOKEN": " "}).suno_url is None


def test_settings_from_env_overrides():
    s = Settings.from_env({
        "COMFY_URL_FLUX": "http://x:1",
        "COMFY_URL_VIDEO": "http://video:2",
        "EASEL_API_KEY": "secret",
        "COMFY_JOB_TIMEOUT": "120",
        "COMFY_MAX_INFLIGHT": "2",
        "EASEL_DEFAULT_RESPONSE_FORMAT": "url",
        "EASEL_DEFAULT_STEPS": "6",
        "EASEL_VARIATION_PROMPT": "vary it",
        "EASEL_PROMPT_SEPARATOR": ":::",
    })
    assert s.comfy_url == "http://x:1"
    assert s.comfy_video_url == "http://video:2"
    assert s.api_key == "secret"
    assert s.job_timeout == 120
    assert s.max_inflight == 2
    assert s.default_response_format == "url"
    assert s.default_steps == 6
    assert s.variation_prompt == "vary it"
    assert s.prompt_separator == ":::"


def test_settings_blank_api_key_is_none():
    assert Settings.from_env({"EASEL_API_KEY": ""}).api_key is None
    assert Settings.from_env({"EASEL_API_KEY": "   "}).api_key is None


def test_image_job_directory_override_and_blank_default():
    default = Settings.from_env({}).image_job_dir
    assert Settings.from_env({"EASEL_JOB_DIR": "/private/jobs"}).image_job_dir == "/private/jobs"
    assert Settings.from_env({"EASEL_JOB_DIR": "  "}).image_job_dir == default


def test_image_backend_url_precedence_and_legacy_compatibility():
    assert Settings.from_env({"COMFY_URL_IMAGE": " http://image:1 ",
                              "COMFY_URL_FLUX": "http://legacy:2"}).comfy_url == "http://image:1"
    assert Settings.from_env({"COMFY_URL_IMAGE": " ",
                              "COMFY_URL_FLUX": "http://legacy:2"}).comfy_url == "http://legacy:2"
    assert Settings.from_env({"COMFY_URL_IMAGE": "", "COMFY_URL_FLUX": " "}).comfy_url == \
        Settings.from_env({}).comfy_url
