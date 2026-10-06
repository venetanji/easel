"""Opt-in, serial video matrix with durable receipts and decoded-media checks."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

import httpx

MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024


def plan_case(value: str, capabilities: dict) -> dict:
    match = re.fullmatch(r"([0-9]{3,4}x[0-9]{3,4}):([0-9]{1,3})", value)
    if not match:
        raise ValueError("cases must use SIZE:SECONDS for LTX or SIZE:FRAMES for H3")
    constraints = capabilities.get("size_constraints") or {}
    if not constraints.get("custom_sizes"):
        raise ValueError("server must expose video size_constraints; deploy the sizing update first")
    size = constraints.get("aliases", {}).get(match[1], match[1])
    width, height = (int(value) for value in size.split("x"))
    if any(not constraints["min_dimension"] <= dimension <= constraints["max_dimension"] or
           dimension % constraints["dimension_multiple"] for dimension in (width, height)):
        raise ValueError(f"{size} violates the server's dimension constraints")
    amount = int(match[2])
    if capabilities["model"] == "ltx-2.5":
        control = "seconds"
        limits = capabilities[control]
        frames = amount * capabilities["fps"] + 1
    else:
        control = "frames"
        limits = capabilities[control]
        frames = amount
        if (frames - limits["offset"]) % limits["step"]:
            raise ValueError("H3 frame count is not on the server's temporal grid")
    if not limits["min"] <= amount <= limits["max"]:
        raise ValueError(f"{control} is outside the server's range")
    if width * height > constraints["max_pixels"] or width * height * frames > constraints["max_pixel_frames"]:
        raise ValueError(f"{value} exceeds the server's video budget")
    return {"size": size, control: amount, "expected_frames": frames, "fps": capabilities["fps"],
            "width": width, "height": height, "status": "planned"}


def save_report(directory: Path, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False, encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    Path(output.name).replace(directory / "report.json")


def sample_memory(gpu_index: str | None) -> int | None:
    if gpu_index is None:
        return None
    result = subprocess.run([
        "nvidia-smi", "-i", gpu_index, "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10)
    return int(result.stdout.strip()) if result.returncode == 0 else None


def gpu_snapshot(gpu_index: str) -> dict:
    result = subprocess.run([
        "nvidia-smi", "-i", gpu_index, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10, check=True)
    name, total, driver = (value.strip() for value in result.stdout.strip().split(","))
    return {"index": gpu_index, "name": name, "total_mib": int(total), "driver_version": driver}


def verify_video(path: Path, case: dict) -> dict:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=120, check=True)
    media = json.loads(probe.stdout)
    video = next(stream for stream in media["streams"] if stream["codec_type"] == "video")
    audio = [stream for stream in media["streams"] if stream["codec_type"] == "audio"]
    actual = (video["width"], video["height"], int(video["nb_read_frames"]), Fraction(video["avg_frame_rate"]))
    expected = (case["width"], case["height"], case["expected_frames"], Fraction(case["fps"]))
    if actual != expected or not audio:
        raise ValueError(f"decoded video does not match expected dimensions/timing/audio: {actual} != {expected}")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                   capture_output=True, timeout=120, check=True)
    return {"width": video["width"], "height": video["height"], "frames": int(video["nb_read_frames"]),
            "fps": str(Fraction(video["avg_frame_rate"])), "duration": float(media["format"]["duration"]),
            "audio_codec": audio[0]["codec_name"], "full_decode": "passed"}


def run_matrix(client: httpx.Client, directory: Path, report: dict, *, timeout: float,
               poll_interval: float, gpu_index: str | None = None) -> int:
    save_report(directory, report)
    for index, case in enumerate(report["cases"]):
        if case["status"] == "verified":
            continue
        try:
            if not case.get("receipt"):
                if case["status"] != "planned":
                    raise ValueError("previous submission had no receipt; reconcile upstream before retrying")
                case["status"] = "submitting"
                save_report(directory, report)
                fields = {"model": report["model"], "prompt": report["prompt"], "size": case["size"], "seed": "0"}
                fields.update({key: str(case[key]) for key in ("seconds", "frames") if key in case})
                response = client.post("/v1/videos", data=fields)
                response.raise_for_status()
                case["receipt"] = response.json()
                case["status"] = "queued"
                save_report(directory, report)
            video_id = case["receipt"]["id"]
            print(f"Case {index + 1}: {case['size']}, {case['expected_frames']} frames, {video_id}", flush=True)
            deadline = time.monotonic() + timeout
            while True:
                response = client.get("/v1/videos/" + video_id)
                response.raise_for_status()
                case["last_poll"] = response.json()
                case["status"] = case["last_poll"]["status"]
                if case["status"] not in ("queued", "in_progress", "completed", "failed", "cancelled"):
                    raise ValueError("unexpected video status; reconcile the saved receipt before proceeding")
                memory = sample_memory(gpu_index)
                if memory is not None:
                    case["sampled_gpu_peak_mib"] = max(memory, case.get("sampled_gpu_peak_mib", 0))
                save_report(directory, report)
                if case["status"] == "completed":
                    break
                if case["status"] in ("failed", "cancelled"):
                    print(f"Stopped: {case['last_poll'].get('error')}", flush=True)
                    return 1
                if time.monotonic() >= deadline:
                    print("Timed out; receipt saved. Resume this report, do not submit a replacement.", flush=True)
                    return 1
                time.sleep(poll_interval)
            path = directory / f"case-{index + 1:02}.mp4"
            downloaded = 0
            with client.stream("GET", "/v1/videos/" + video_id + "/content") as response:
                response.raise_for_status()
                with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as output:
                    for chunk in response.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > MAX_DOWNLOAD_BYTES:
                            raise ValueError("video exceeds the benchmark download limit")
                        output.write(chunk)
            case["verification"] = verify_video(path, case)
            case["status"] = "verified"
            case["artifact"] = path.name
            save_report(directory, report)
            print(f"Verified {path.name}", flush=True)
        except KeyboardInterrupt:
            save_report(directory, report)
            print("Interrupted; report saved. Resume/reconcile the existing receipt, not a replacement.", flush=True)
            return 1
        except (httpx.HTTPError, ValueError, OSError, subprocess.SubprocessError, KeyError, StopIteration) as error:
            case["benchmark_error"] = str(error)
            save_report(directory, report)
            print(f"Stopped: {type(error).__name__}. Receipt/state saved in {directory / 'report.json'}", flush=True)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("EASEL_URL", "http://127.0.0.1:8799"))
    parser.add_argument("--model", choices=["ltx-2.5", "minimax-h3"])
    parser.add_argument("--case", action="append", default=[], help="SIZE:SECONDS (LTX) or SIZE:FRAMES (H3), at most 8")
    parser.add_argument("--output", type=Path, required=True, help="private artifact directory outside the repository")
    parser.add_argument("--execute", action="store_true", help="submit GPU jobs; otherwise only discover and validate the plan")
    parser.add_argument("--resume", action="store_true", help="reuse saved receipts instead of submitting replacements")
    parser.add_argument("--gpu-index", help="optional local nvidia-smi device index; measures total device usage")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--poll-interval", type=float, default=2)
    args = parser.parse_args(argv)
    url = urlsplit(args.url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("use an HTTP(S) base URL without embedded credentials, query or fragment")
    if args.timeout <= 0 or args.poll_interval <= 0:
        parser.error("timeouts and poll intervals must be positive")
    if args.execute and any(shutil.which(tool) is None for tool in ("ffprobe", "ffmpeg")):
        parser.error("ffprobe and ffmpeg are required before submitting GPU jobs")
    if args.gpu_index is not None and shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi is required for GPU memory sampling")
    report_path = args.output / "report.json"
    if args.resume:
        if args.case:
            parser.error("resume cannot add new cases")
        report = json.loads(report_path.read_text())
        if report["url"] != args.url.rstrip("/") or (args.model and report["model"] != args.model):
            parser.error("resume URL/model must match the saved report")
        model = report["model"]
    else:
        if report_path.exists():
            parser.error("report already exists; use --resume or a new output directory")
        if args.execute and args.output.exists() and any(args.output.iterdir()):
            parser.error("a new benchmark requires an empty output directory")
        if not 1 <= len(args.case) <= 8:
            parser.error("provide 1 through 8 cases")
        model = args.model or "ltx-2.5"
    headers = {"Authorization": "Bearer " + os.environ["EASEL_API_KEY"]} if os.environ.get("EASEL_API_KEY") else {}
    with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=60) as client:
        response = client.get("/v1/videos/capabilities", params={"model": model})
        response.raise_for_status()
        capabilities = response.json()
        if not args.resume:
            try:
                cases = [plan_case(value, capabilities) for value in args.case]
            except ValueError as error:
                parser.error(str(error))
            report = {"schema_version": 1, "url": args.url.rstrip("/"), "model": model,
                      "created_at": int(time.time()), "capabilities": capabilities, "cases": cases,
                      "prompt": "One continuous locked-off shot of a small red wooden ball rolling slowly across a plain table. Quiet room tone, no speech or music."}
            if args.gpu_index is not None:
                report["gpu"] = gpu_snapshot(args.gpu_index)
        if not args.execute:
            print(json.dumps({"dry_run": True, "model": model, "cases": report["cases"]}, indent=2))
            return 0
        return run_matrix(client, args.output, report, timeout=args.timeout,
                          poll_interval=args.poll_interval, gpu_index=args.gpu_index)


if __name__ == "__main__":
    raise SystemExit(main())
