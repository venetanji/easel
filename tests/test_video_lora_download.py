"""Pinned downloads never replace existing or unverified weights."""
import hashlib
import importlib.util
import io
from pathlib import Path
import urllib.error

import pytest


@pytest.fixture
def downloader():
    spec = importlib.util.spec_from_file_location("lora_downloader",
        Path(__file__).parents[1] / "scripts" / "download_video_loras.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset(payload):
    return {"filename": "adapter.safetensors", "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def entry():
    return {"id": "test", "repo_id": "Lightricks/test", "revision": "a" * 40}


def test_download_verifies_checksum_before_install(downloader, tmp_path, monkeypatch):
    payload = b"weights"
    def open_response(request, timeout):
        assert "/resolve/" + "a" * 40 + "/adapter.safetensors" in request.full_url
        assert request.get_header("Authorization") == "Bearer secret"
        return io.BytesIO(payload)
    monkeypatch.setattr(downloader, "open_download", open_response)
    outcome = downloader.install_asset(entry(), asset(payload), tmp_path, "secret")
    assert outcome["status"] == "installed"
    assert (tmp_path / "adapter.safetensors").read_bytes() == payload
    assert list(tmp_path.iterdir()) == [tmp_path / "adapter.safetensors"]


def test_failed_checksum_leaves_no_weights_or_partial_file(downloader, tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "open_download", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="checksum/size"):
        downloader.install_asset(entry(), asset(b"weights"), tmp_path, None)
    assert list(tmp_path.iterdir()) == []


def test_existing_different_weights_are_not_overwritten(downloader, tmp_path):
    target = tmp_path / "adapter.safetensors"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="not overwritten"):
        downloader.install_asset(entry(), asset(b"weights"), tmp_path, None)
    assert target.read_bytes() == b"existing"


def test_verified_existing_weights_need_no_network(downloader, tmp_path, monkeypatch):
    target = tmp_path / "adapter.safetensors"
    target.write_bytes(b"weights")
    monkeypatch.setattr(downloader, "open_download", lambda *args, **kwargs: pytest.fail("network"))
    assert downloader.install_asset(entry(), asset(b"weights"), tmp_path, None)["status"] == "already_verified"


def test_gated_download_reports_access_without_accepting_terms(downloader, tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise urllib.error.HTTPError("url", 403, "gated", {}, None)
    monkeypatch.setattr(downloader, "open_download", denied)
    outcome = downloader.install_asset(entry(), asset(b"weights"), tmp_path, None)
    assert outcome["status"] == "access_denied"
    assert outcome["http_status"] == 403
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_path_traversal(downloader, tmp_path):
    unsafe = {**asset(b"weights"), "filename": "../adapter.safetensors"}
    with pytest.raises(ValueError, match="unsafe"):
        downloader.install_asset(entry(), unsafe, tmp_path, None)


def test_redirect_does_not_leak_hf_token_to_cdn(downloader):
    request = downloader.urllib.request.Request("https://huggingface.co/weights",
        headers={"Authorization": "Bearer secret"})
    redirected = downloader.HFRedirectHandler().redirect_request(request, None, 302,
        "Found", {}, "https://cdn.example/weights?signature=test")
    assert redirected.get_header("Authorization") is None


def test_same_host_redirect_preserves_authentication(downloader):
    request = downloader.urllib.request.Request("https://huggingface.co/weights",
        headers={"Authorization": "Bearer secret"})
    redirected = downloader.HFRedirectHandler().redirect_request(request, None, 302,
        "Found", {}, "https://huggingface.co/other-weights")
    assert redirected.get_header("Authorization") == "Bearer secret"
