"""Unit tests for the ComfyUI client, using httpx.MockTransport (no real network)."""
import json

import httpx
import pytest

from easel.comfy_client import (
    ComfyClient,
    ComfyError,
    ComfyExecError,
    ComfySubmitError,
    ComfyTimeout,
)


def make_client(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ComfyClient("http://comfy", http)


# ---- upload_image ----

async def test_upload_image_returns_server_name_and_subfolder():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["content"] = request.content
        return httpx.Response(200, json={"name": "easel_renamed.png", "subfolder": "", "type": "input"})

    client = make_client(handler)
    name, subfolder = await client.upload_image(b"PNGDATA", "easel_orig.png")
    assert name == "easel_renamed.png"  # trust the server's name, not ours
    assert subfolder == ""
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/upload/image")
    assert b"overwrite" in seen["content"] and b"true" in seen["content"]


# ---- submit ----

async def test_submit_returns_prompt_id_and_sends_client_id():
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["body"] = body
        return httpx.Response(200, json={"prompt_id": "pid-1", "number": 1})

    client = make_client(handler)
    pid = await client.submit({"1": {"class_type": "X", "inputs": {}}})
    assert pid == "pid-1"
    assert "prompt" in seen["body"] and "client_id" in seen["body"]


async def test_submit_400_with_node_errors_raises_submit_error():
    def handler(request):
        return httpx.Response(400, json={
            "error": {"message": "invalid prompt"},
            "node_errors": {"3": {"errors": [{"message": "Required input is missing"}]}},
        })

    client = make_client(handler)
    with pytest.raises(ComfySubmitError) as ei:
        await client.submit({"3": {"class_type": "X", "inputs": {}}})
    assert "3" in ei.value.node_errors
    assert "missing" in str(ei.value).lower()


async def test_submit_200_without_prompt_id_raises_submit_error():
    def handler(request):
        return httpx.Response(200, json={"node_errors": {"1": {}}, "error": "bad"})

    client = make_client(handler)
    with pytest.raises(ComfySubmitError):
        await client.submit({"1": {"class_type": "X", "inputs": {}}})


# ---- wait ----

def _history(prompt_id, outputs, status_str="success", messages=None):
    return {prompt_id: {
        "outputs": outputs,
        "status": {"status_str": status_str, "completed": status_str == "success",
                   "messages": messages or []},
    }}


async def test_wait_collects_images_across_all_nodes_and_entries():
    outputs = {
        "9": {"images": [
            {"filename": "a.png", "subfolder": "", "type": "output"},
            {"filename": "b.png", "subfolder": "", "type": "output"},
        ]},
        "10": {"images": [{"filename": "c.png", "subfolder": "sub", "type": "output"}]},
    }

    def handler(request):
        return httpx.Response(200, json=_history("pid-1", outputs))

    client = make_client(handler)
    refs = await client.wait("pid-1", timeout=5, poll_interval=0.001)
    assert sorted(r["filename"] for r in refs) == ["a.png", "b.png", "c.png"]
    assert any(r["subfolder"] == "sub" for r in refs)


async def test_wait_polls_until_history_present():
    calls = {"n": 0}
    ready = _history("pid-1", {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}})

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json={})  # not ready yet
        return httpx.Response(200, json=ready)

    client = make_client(handler)
    refs = await client.wait("pid-1", timeout=5, poll_interval=0.001)
    assert len(refs) == 1
    assert calls["n"] >= 3


async def test_wait_timeout_raises_with_prompt_id():
    def handler(request):
        return httpx.Response(200, json={})  # never ready

    client = make_client(handler)
    with pytest.raises(ComfyTimeout) as ei:
        await client.wait("pid-9", timeout=0.05, poll_interval=0.01)
    assert ei.value.prompt_id == "pid-9"


async def test_wait_execution_error_raises_exec_error_with_details():
    messages = [
        ["execution_start", {}],
        ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}],
    ]

    def handler(request):
        return httpx.Response(200, json=_history("pid-1", {}, status_str="error", messages=messages))

    client = make_client(handler)
    with pytest.raises(ComfyExecError) as ei:
        await client.wait("pid-1", timeout=5, poll_interval=0.01)
    assert ei.value.node_type == "KSampler"
    assert "CUDA out of memory" in str(ei.value)


# ---- fetch ----

async def test_fetch_returns_bytes_with_view_params():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"PNGBYTES")

    client = make_client(handler)
    data = await client.fetch({"filename": "a.png", "subfolder": "sub", "type": "output"})
    assert data == b"PNGBYTES"
    assert "filename=a.png" in seen["url"]
    assert "subfolder=sub" in seen["url"]
    assert "type=output" in seen["url"]


async def test_transport_error_surfaces_as_comfy_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = make_client(handler)
    with pytest.raises((ComfyError, httpx.HTTPError)):
        await client.submit({"1": {"class_type": "X", "inputs": {}}})
