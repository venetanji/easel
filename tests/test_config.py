"""Unit tests for the model table, size parsing, and settings."""
import pytest

from easel.config import MODELS, resolve_model, parse_size, UnknownModelError, Settings


# ---- model table ----

def test_resolve_known_models():
    nine = resolve_model("flux2-9b")
    assert nine.unet == "flux-2-klein-9b-fp8.safetensors"
    assert nine.clip == "qwen_3_8b_fp8mixed.safetensors"
    four = resolve_model("flux2-4b")
    assert four.unet == "flux-2-klein-4b-fp8.safetensors"
    # klein-4b needs the 4b text encoder (7680-dim), not the 8b one
    assert four.clip == "qwen_3_4b_fp4_flux2.safetensors"


def test_model_table_ids():
    assert set(MODELS) == {"flux2-9b", "flux2-4b"}


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
    assert s.api_key is None
    assert s.job_timeout == 600
    assert s.max_inflight == 1
    assert s.default_response_format == "b64_json"
    assert s.default_steps == 8
    assert s.variation_prompt


def test_settings_from_env_overrides():
    s = Settings.from_env({
        "COMFY_URL_FLUX": "http://x:1",
        "EASEL_API_KEY": "secret",
        "COMFY_JOB_TIMEOUT": "120",
        "COMFY_MAX_INFLIGHT": "2",
        "EASEL_DEFAULT_RESPONSE_FORMAT": "url",
        "EASEL_DEFAULT_STEPS": "6",
        "EASEL_VARIATION_PROMPT": "vary it",
    })
    assert s.comfy_url == "http://x:1"
    assert s.api_key == "secret"
    assert s.job_timeout == 120
    assert s.max_inflight == 2
    assert s.default_response_format == "url"
    assert s.default_steps == 6
    assert s.variation_prompt == "vary it"


def test_settings_blank_api_key_is_none():
    assert Settings.from_env({"EASEL_API_KEY": ""}).api_key is None
    assert Settings.from_env({"EASEL_API_KEY": "   "}).api_key is None
