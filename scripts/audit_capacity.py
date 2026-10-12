"""Opt-in real generation burst; save receipts before polling, never retry POSTs."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(json.dumps(report, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


async def audit(options) -> None:
    load_dotenv()
    key = os.environ.get("EASEL_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    path = Path(options.output)
    if options.resume:
        report = json.loads(path.read_text())
        if report["base_url"] != options.url.rstrip("/"):
            raise SystemExit("Resume URL must match the saved report.")
    else:
        if not options.live:
            raise SystemExit("Pass --live to spend credits and submit real GPU jobs.")
        if path.exists():
            raise SystemExit("Report already exists; use --resume or choose a new output.")
        report = {"started_at": datetime.now(timezone.utc).isoformat(),
                  "base_url": options.url.rstrip("/"), "requests": [], "health": [],
                  "parameters": {"images": options.images, "videos": options.videos,
                                 "audio": options.audio, "image_size": options.image_size,
                                 "image_steps": options.image_steps,
                                 "video_size": options.video_size,
                                 "video_seconds": options.video_seconds}}
        save_report(path, report)
    async with httpx.AsyncClient(base_url=report["base_url"], headers=headers,
                                timeout=190, limits=httpx.Limits(max_connections=40)) as client:
        stop = asyncio.Event()

        async def health_monitor():
            while not stop.is_set():
                started = time.monotonic()
                try:
                    response = await client.get("/health", timeout=5)
                    sample = {"http_status": response.status_code}
                except httpx.HTTPError as error:
                    sample = {"error": type(error).__name__}
                sample["seconds"] = round(time.monotonic() - started, 3)
                report["health"].append(sample)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2)
                except TimeoutError:
                    pass

        monitor = asyncio.create_task(health_monitor())
        try:
            if not options.resume:
                response = await client.get("/v1/audio/generations/status") if options.audio else None
                if response is not None:
                    response.raise_for_status()
                    state = response.json()
                    report["audio_before"] = {name: state.get(name) for name in
                                              ("status", "attempt_id", "recovery_required")}
                    if state.get("status") in {"pending", "captcha_required"}:
                        raise SystemExit("Suno has an unresolved attempt; no new Create submitted.")

                async def submit(kind: str, index: int):
                    started = time.monotonic()
                    item = {"kind": kind, "index": index,
                            "submitted_at": datetime.now(timezone.utc).isoformat(),
                            "submission_uncertain": True}
                    report["requests"].append(item)
                    save_report(path, report)
                    try:
                        if kind == "image":
                            response = await client.post("/v1/images/generations",
                                headers={"Prefer": "respond-async"}, json={
                                    "model": "qwen-image-2.1", "prompt":
                                    f"Studio photograph of a red wooden chair, capacity audit {index}",
                                    "size": options.image_size, "steps": options.image_steps,
                                    "seed": 9000 + index, "response_format": "url"})
                        elif kind == "video":
                            response = await client.post("/v1/videos", data={
                                "model": "ltx-2.5", "prompt":
                                f"A red wooden chair slowly turning in a bright studio, audit {index}",
                                "size": options.video_size, "seconds": options.video_seconds,
                                "seed": 10000 + index})
                        else:
                            response = await client.post("/v1/audio/generations", json={
                                "model": "suno-music", "prompt":
                                "Short mellow acoustic instrumental, gentle guitar and light percussion",
                                "title": f"Easel capacity audit {index}",
                                "make_instrumental": True, "duration_seconds": 30})
                        item.update(http_status=response.status_code,
                                    retry_after=response.headers.get("Retry-After"))
                        try:
                            body = response.json()
                            if isinstance(body, dict):
                                item["receipt"] = body
                                has_handle = bool(body.get("id") or body.get("attempt_id")
                                                  or body.get("songs"))
                                item["submission_uncertain"] = response.status_code >= 500 or (
                                    response.is_success and not has_handle)
                            else:
                                item["body"] = response.text[:500]
                        except ValueError:
                            item["body"] = response.text[:500]
                    except httpx.HTTPError as error:
                        item["error"] = type(error).__name__
                        item["submission_uncertain"] = True
                    item["acceptance_seconds"] = round(time.monotonic() - started, 3)
                    save_report(path, report)
                    print(json.dumps({name: item.get(name) for name in
                                      ("kind", "index", "http_status", "acceptance_seconds", "error")}),
                          flush=True)

                await asyncio.gather(*(submit(kind, index) for kind, count in
                    (("image", options.images), ("video", options.videos), ("audio", options.audio))
                    for index in range(count)))
                report["submissions_complete_at"] = datetime.now(timezone.utc).isoformat()
                save_report(path, report)
                print("All submission responses saved. POSTs will never be retried.", flush=True)
                if options.submit_only:
                    return

            deadline = time.monotonic() + options.poll_seconds
            pending = [item for item in report["requests"] if item.get("http_status") in (200, 202)
                       and item.get("receipt", {}).get("id") and item["kind"] in {"image", "video"}
                       and item.get("final", {}).get("status") not in {"completed", "failed", "cancelled"}]
            while pending and time.monotonic() < deadline:
                for item in pending[:]:
                    job_id = item["receipt"]["id"]
                    route = "/v1/images/jobs/" if item["kind"] == "image" else "/v1/videos/"
                    try:
                        response = await client.get(route + job_id, timeout=30)
                        body = response.json()
                        item["last_poll_http_status"] = response.status_code
                        item["final"] = body
                        if response.status_code == 200 and body.get("status") in {
                            "completed", "failed", "cancelled"
                        }:
                            item["observed_terminal_at"] = datetime.now(timezone.utc).isoformat()
                            if body["status"] == "completed":
                                content_route = (route + job_id + "/content/0" if item["kind"] == "image"
                                                 else route + job_id + "/content")
                                media = await client.get(content_route, timeout=60)
                                item["output"] = {"http_status": media.status_code,
                                                  "bytes": len(media.content),
                                                  "content_type": media.headers.get("content-type")}
                            pending.remove(item)
                            print(f"{item['kind']} {item['index']}: {body['status']}", flush=True)
                    except (httpx.HTTPError, ValueError) as error:
                        item["last_poll_error"] = type(error).__name__
                save_report(path, report)
                if pending:
                    await asyncio.sleep(3)

            audio_items = [item for item in report["requests"] if item["kind"] == "audio"]
            audio_pending = [item for item in audio_items if item.get("receipt", {}).get("songs")]
            while audio_pending and time.monotonic() < deadline:
                for item in audio_pending[:]:
                    tracks = []
                    for song in item["receipt"]["songs"]:
                        try:
                            response = await client.get("/v1/audio/tracks/" + song["id"], timeout=150)
                            tracks.append({"http_status": response.status_code, "body": response.json()})
                        except (httpx.HTTPError, ValueError) as error:
                            tracks.append({"error": type(error).__name__})
                    item["tracks"] = tracks
                    if all(track.get("http_status") == 200 and
                           track.get("body", {}).get("status") in {"complete", "error"}
                           for track in tracks):
                        audio_pending.remove(item)
                save_report(path, report)
                if audio_pending:
                    await asyncio.sleep(3)
            if audio_items:
                response = await client.get("/v1/audio/generations/status", timeout=150)
                report["audio_after"] = response.json()
            report["polling_finished_at"] = datetime.now(timezone.utc).isoformat()
            report["unfinished_gpu_jobs"] = [item["receipt"]["id"] for item in pending]
            report["unfinished_audio_attempts"] = [item["receipt"].get("attempt_id")
                                                   for item in audio_pending]
        finally:
            stop.set()
            await monitor
            save_report(path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8799")
    parser.add_argument("--output", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--videos", type=int, default=12)
    parser.add_argument("--audio", type=int, default=2)
    parser.add_argument("--image-size", default="512x512")
    parser.add_argument("--image-steps", type=int, default=25)
    parser.add_argument("--video-size", default="512x320")
    parser.add_argument("--video-seconds", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=900)
    options = parser.parse_args()
    if not all(0 <= count <= 32 for count in (options.images, options.videos, options.audio)):
        parser.error("Each burst count must be between 0 and 32.")
    asyncio.run(audit(options))


if __name__ == "__main__":
    main()
