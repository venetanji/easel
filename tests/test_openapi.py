"""Swagger metadata and auth regression tests; no external services or real keys."""
import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easel.app import create_app
from easel.config import Settings


@pytest.fixture
def client(tmp_path):
    settings = dataclasses.replace(Settings.from_env({}), api_key="test-only-key",
                                   image_job_dir=str(tmp_path))
    return TestClient(create_app(settings=settings))


def request_schema(schema, endpoint, content_type):
    reference = schema["paths"][endpoint]["post"]["requestBody"]["content"][content_type]["schema"]["$ref"]
    return schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]


def test_openapi_advertises_bearer_on_every_protected_operation(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["components"]["securitySchemes"]["BearerAuth"]["type"] == "http"
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    assert not schema.get("security")
    for path, operations in schema["paths"].items():
        for operation in operations.values():
            if path == "/health":
                assert not operation.get("security")
            else:
                assert path.startswith("/v1/")
                assert operation["security"] == [{"BearerAuth": []}], path
    assert "test-only-key" not in response.text


def test_all_routes_have_ordered_groups_and_usage_descriptions(client):
    schema = client.get("/openapi.json").json()
    assert [tag["name"] for tag in schema["tags"]] == [
        "Health", "Models", "Images", "Videos", "Audio"]
    for path, operations in schema["paths"].items():
        expected = "Health" if path == "/health" else path.split("/")[2].title()
        for operation in operations.values():
            assert operation["tags"] == [expected], path
            assert operation["summary"]
            assert operation["description"], path


@pytest.mark.parametrize("path", ["/health", "/docs", "/redoc", "/openapi.json"])
def test_public_routes_remain_public(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("authorization", [None, "", "Bearer", "Bearer ",
    "Basic test-only-key", "Bearer incorrect", "Bearer test-only-key extra", "Bearer\ttest-only-key"])
def test_models_preserves_exact_auth_error(client, authorization):
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.get("/v1/models", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": {"message": "invalid or missing API key",
        "type": "invalid_request_error", "code": "invalid_api_key", "param": None}}


@pytest.mark.parametrize("authorization", ["Bearer test-only-key", "bEaReR test-only-key",
                                         "Bearer   test-only-key  "])
def test_models_accepts_existing_bearer_variations(client, authorization):
    assert client.get("/v1/models", headers={"Authorization": authorization}).status_code == 200


@pytest.mark.parametrize("authorization", [None, "Basic ignored", "Bearer wrong", "Bearer"])
def test_unconfigured_key_keeps_auth_optional(tmp_path, authorization):
    settings = dataclasses.replace(Settings.from_env({}), image_job_dir=str(tmp_path))
    client = TestClient(create_app(settings=settings))
    headers = {} if authorization is None else {"Authorization": authorization}
    assert client.get("/v1/models", headers=headers).status_code == 200


def test_query_key_is_not_authentication(client):
    assert client.get("/v1/models?api_key=test-only-key&token=test-only-key").status_code == 401


def test_swagger_uses_openapi_without_persisting_credentials(client):
    docs = client.get("/docs").text
    assert "SwaggerUIBundle" in docs
    assert "/openapi.json" in docs
    assert "persistAuthorization" not in docs
    assert "test-only-key" not in docs


def test_every_protected_route_still_rejects_missing_credentials(client):
    schema = client.get("/openapi.json").json()
    for path, operations in schema["paths"].items():
        if not path.startswith("/v1/"):
            continue
        url = (path.replace("{video_id}", "unused").replace("{job_id}", "unused")
               .replace("{index}", "0").replace("{song_id}", "11111111-1111-4111-8111-111111111111"))
        for method in operations:
            kwargs = {}
            if path == "/v1/images/view":
                kwargs["params"] = {"filename": "unused.png"}
            if method == "post":
                if path == "/v1/videos":
                    kwargs["data"] = {"model": "ltx-2.5", "prompt": "test"}
                else:
                    kwargs["json"] = {"model": "suno-sound", "prompt": "test"}
                    if path.endswith("/abandon"):
                        kwargs["json"] = {"attempt_id": "11111111-1111-4111-8111-111111111111"}
            response = client.request(method, url, **kwargs)
            assert response.status_code == 401, (method, path, response.text)
            assert response.json()["error"]["code"] == "invalid_api_key", path


def test_image_auth_still_precedes_manual_json_validation(client):
    response = client.post("/v1/images/generations", content="not JSON")
    assert response.status_code == 401
    response = client.post("/v1/images/generations", content="not JSON",
                           headers={"Authorization": "Bearer test-only-key"})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "request body must be valid JSON"


def test_image_swagger_bodies_and_async_header_follow_existing_contract(client):
    from easel.config import MODELS
    from easel.qwen_graph import MAX_REFERENCES

    schema = client.get("/openapi.json").json()
    for endpoint in ("generations", "edits", "variations"):
        operation = schema["paths"][f"/v1/images/{endpoint}"]["post"]
        prefer = next(p for p in operation["parameters"] if p["name"] == "Prefer")
        assert prefer["in"] == "header"
        assert not prefer["required"]
        media_type = "application/json" if endpoint == "generations" else "multipart/form-data"
        body = request_schema(schema, f"/v1/images/{endpoint}", media_type)
        assert body["properties"]["model"]["enum"] == list(MODELS)
        if endpoint == "generations":
            assert body["required"] == ["prompt"]
        elif endpoint == "edits":
            assert body["properties"]["image"]["maxItems"] == MAX_REFERENCES
            assert body["properties"]["image"]["items"]["format"] == "binary"
        else:
            assert body["properties"]["image"]["format"] == "binary"


def test_audio_examples_match_real_request_models_without_replacing_schema(client):
    from pydantic import TypeAdapter
    from easel.audio_api import MusicGeneration, SpeechGeneration, SoundGeneration

    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/audio/generations"]["post"]
    media = operation["requestBody"]["content"]["application/json"]
    assert len(request_schema(schema, "/v1/audio/generations", "application/json")["anyOf"]) == 3
    adapter = TypeAdapter(MusicGeneration | SpeechGeneration | SoundGeneration)
    assert set(media["examples"]) == {"music", "speech", "sound"}
    for example in media["examples"].values():
        validated = adapter.validate_python(example["value"])
        assert validated.dry_run is True


def test_docs_preserve_network_auth_policy_and_native_audio_discovery(client):
    schema = client.get("/openapi.json").json()
    description = schema["info"]["description"]
    assert "EASEL_TRUSTED_CLIENT_CIDRS" in description
    assert "EASEL_REQUIRE_AUTH_CIDRS" in description
    operation = schema["paths"]["/v1/audio/models"]["get"]
    assert operation["tags"] == ["Audio"]
    assert operation["security"] == [{"BearerAuth": []}]
    assert "suno_model" in operation["description"]
    assert "default_model" in operation["description"]


def test_all_component_schemas_have_clear_consistent_names(client):
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    assert set(components) == {
        "ImageGenerationRequest", "ImageEditRequest", "ImageVariationRequest",
        "VideoGenerationRequest", "AudioGenerationRequest",
        "AudioMusicGenerationRequest", "AudioSpeechGenerationRequest",
        "AudioSoundGenerationRequest", "AudioGenerationAbandonRequest",
        "ValidationErrorResponse", "ValidationErrorDetail",
    }
    assert list(components) == sorted(components)
    for name, definition in components.items():
        assert definition["title"] == name


@pytest.mark.parametrize("endpoint,content_type,name", [
    ("/v1/images/generations", "application/json", "ImageGenerationRequest"),
    ("/v1/images/edits", "multipart/form-data", "ImageEditRequest"),
    ("/v1/images/variations", "multipart/form-data", "ImageVariationRequest"),
    ("/v1/videos", "multipart/form-data", "VideoGenerationRequest"),
    ("/v1/audio/generations", "application/json", "AudioGenerationRequest"),
    ("/v1/audio/generations/abandon", "application/json", "AudioGenerationAbandonRequest"),
])
def test_every_request_body_uses_a_named_schema(client, endpoint, content_type, name):
    schema = client.get("/openapi.json").json()
    media = schema["paths"][endpoint]["post"]["requestBody"]["content"][content_type]
    assert media["schema"] == {"$ref": "#/components/schemas/" + name}


def test_schema_renaming_preserves_all_references_and_audio_variants(client):
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    def check_references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].startswith("#/components/schemas/")
                assert value["$ref"].rsplit("/", 1)[-1] in components
            for nested in value.values():
                check_references(nested)
        elif isinstance(value, list):
            for nested in value:
                check_references(nested)

    check_references(schema)
    assert components["AudioGenerationRequest"]["anyOf"] == [
        {"$ref": "#/components/schemas/AudioMusicGenerationRequest"},
        {"$ref": "#/components/schemas/AudioSpeechGenerationRequest"},
        {"$ref": "#/components/schemas/AudioSoundGenerationRequest"},
    ]


def test_named_openapi_schema_caches_and_regenerates_cleanly(client):
    original = client.app.openapi()
    assert client.app.openapi() is original
    client.app.openapi_schema = None
    regenerated = client.app.openapi()
    assert regenerated == original
    assert regenerated is not original


@pytest.mark.parametrize("model_name,schema_name", [
    ("MusicGeneration", "AudioMusicGenerationRequest"),
    ("SpeechGeneration", "AudioSpeechGenerationRequest"),
    ("SoundGeneration", "AudioSoundGenerationRequest"),
    ("AbandonGeneration", "AudioGenerationAbandonRequest"),
])
def test_named_audio_schemas_preserve_generated_request_definitions(client, model_name, schema_name):
    named = client.app.openapi()["components"]["schemas"][schema_name]
    client.app.openapi_schema = None
    original = FastAPI.openapi(client.app)
    expected = original["components"]["schemas"][model_name].copy()
    expected["title"] = schema_name
    assert named == expected


def test_named_video_schema_preserves_generated_form_definition(client):
    named = client.app.openapi()["components"]["schemas"]["VideoGenerationRequest"].copy()
    client.app.openapi_schema = None
    original = FastAPI.openapi(client.app)
    reference = original["paths"]["/v1/videos"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    generated = original["components"]["schemas"][reference.rsplit("/", 1)[-1]].copy()
    named.pop("title")
    generated.pop("title")
    assert named == generated
